from pathlib import Path
from typing import Union, List
import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from sklearn.metrics import roc_curve
# from time import perf_counter as timer
import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_

from resemblyzer.hparams import (
    sampling_rate, partials_n_frames,
    mel_n_channels, mel_window_step,
    model_hidden_size, model_embedding_size, model_num_layers,
    learning_rate_init, speakers_per_batch, utterances_per_speaker
)
from resemblyzer import audio
from resemblyzer.data_objects import SpeakerVerificationDataLoader, SpeakerVerificationDataset
from utils.logger import Profiler, Logger
# from resemblyzer.transformer import TransformerEncoder


class VoiceEncoder(nn.Module):
    def __init__(self,
                 device: Union[str, torch.device] = None,
                 loss_device='cpu'):
        """
        :param device: either a torch device or the name of a torch device (e.g. "cpu", "cuda").
        If None, defaults to cuda if it is available on your machine, otherwise the model will
        run on cpu. Outputs are always returned on the cpu, as numpy arrays.
        """
        super().__init__()

        # Get the target device
        if device is None:
            device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            device = torch.device(device)
        self.device = device

        if isinstance(loss_device, str):
            loss_device = torch.device(loss_device)
        self.loss_device = loss_device

        # Define the network
        self.lstm = nn.LSTM(mel_n_channels,
                            model_hidden_size,
                            model_num_layers,
                            batch_first=True).to(device)
        self.linear = nn.Linear(model_hidden_size,
                                model_embedding_size).to(device)
        self.relu = nn.ReLU().to(device)

        # self.transformer = TransformerEncoder(mel_n_channels, model_num_layers, trans_heads, model_hidden_size, partials_n_frames).to(device)

        # Cosine similarity scaling (with fixed initial parameter values)
        self.similarity_weight = nn.Parameter(torch.tensor([10.
                                                            ])).to(loss_device)
        self.similarity_bias = nn.Parameter(torch.tensor([-5.
                                                          ])).to(loss_device)
        # Loss
        self.loss_fn = nn.CrossEntropyLoss().to(loss_device)

    def forward(self, mels: torch.FloatTensor):
        """
        Computes the embeddings of a batch of utterance spectrograms.

        :param mels: a batch of mel spectrograms of same duration as a float32 tensor of shape
        (batch_size, n_frames, n_channels)
        :return: the embeddings as a float 32 tensor of shape (batch_size, embedding_size).
        Embeddings are positive and L2-normed, thus they lay in the range [0, 1].
        """
        # Pass the input through the LSTM layers and retrieve the final hidden state of the last
        # layer. Apply a cutoff to 0 for negative values and L2 normalize the embeddings.

        _, (hidden, _) = self.lstm(mels)
        embeds_raw = self.relu(self.linear(hidden[-1]))
        return embeds_raw / torch.norm(embeds_raw, dim=1, keepdim=True)

    def load_ckpt(self,
                  ckpt_path='ckpt/pretrained.pt',
                  device=torch.device('cuda')):
        encoder_ckpt = torch.load(ckpt_path, map_location="cpu")
        self.load_state_dict(encoder_ckpt["model_state"], strict=False)
        self.to(device)

    @staticmethod
    def compute_partial_slices(n_samples: int, rate, min_coverage):
        """
        Computes where to split an utterance waveform and its corresponding mel spectrogram to
        obtain partial utterances of <partials_n_frames> each. Both the waveform and the
        mel spectrogram slices are returned, so as to make each partial utterance waveform
        correspond to its spectrogram.

        The returned ranges may be indexing further than the length of the waveform. It is
        recommended that you pad the waveform with zeros up to wav_slices[-1].stop.

        :param n_samples: the number of samples in the waveform
        :param rate: how many partial utterances should occur per second. Partial utterances must
        cover the span of the entire utterance, thus the rate should not be lower than the inverse
        of the duration of a partial utterance. By default, partial utterances are 1.6s long and
        the minimum rate is thus 0.625.
        :param min_coverage: when reaching the last partial utterance, it may or may not have
        enough frames. If at least <min_pad_coverage> of <partials_n_frames> are present,
        then the last partial utterance will be considered by zero-padding the audio. Otherwise,
        it will be discarded. If there aren't enough frames for one partial utterance,
        this parameter is ignored so that the function always returns at least one slice.
        :return: the waveform slices and mel spectrogram slices as lists of array slices. Index
        respectively the waveform and the mel spectrogram with these slices to obtain the partial
        utterances.
        """
        assert 0 < min_coverage <= 1

        # Compute how many frames separate two partial utterances
        samples_per_frame = int((sampling_rate * mel_window_step / 1000))
        n_frames = int(np.ceil((n_samples + 1) / samples_per_frame))
        frame_step = int(np.round((sampling_rate / rate) / samples_per_frame))
        assert 0 < frame_step, "The rate is too high"
        assert frame_step <= partials_n_frames, "The rate is too low, it should be %f at least" % \
            (sampling_rate / (samples_per_frame * partials_n_frames))

        # Compute the slices
        wav_slices, mel_slices = [], []
        steps = max(1, n_frames - partials_n_frames + frame_step + 1)
        for i in range(0, steps, frame_step):
            mel_range = np.array([i, i + partials_n_frames])
            wav_range = mel_range * samples_per_frame
            mel_slices.append(slice(*mel_range))
            wav_slices.append(slice(*wav_range))

        # Evaluate whether extra padding is warranted or not
        last_wav_range = wav_slices[-1]
        coverage = (n_samples - last_wav_range.start) / (last_wav_range.stop - last_wav_range.start)
        if coverage < min_coverage and len(mel_slices) > 1:
            mel_slices = mel_slices[:-1]
            wav_slices = wav_slices[:-1]

        return wav_slices, mel_slices

    def embed_utterance(self,
                        wav: np.ndarray,
                        return_partials=False,
                        rate=1.3,
                        min_coverage=0.75):
        """
        Computes an embedding for a single utterance. The utterance is divided in partial
        utterances and an embedding is computed for each. The complete utterance embedding is the
        L2-normed average embedding of the partial utterances.

        TODO: independent batched version of this function

        :param wav: a preprocessed utterance waveform as a numpy array of float32
        :param return_partials: if True, the partial embeddings will also be returned along with
        the wav slices corresponding to each partial utterance.
        :param rate: how many partial utterances should occur per second. Partial utterances must
        cover the span of the entire utterance, thus the rate should not be lower than the inverse
        of the duration of a partial utterance. By default, partial utterances are 1.6s long and
        the minimum rate is thus 0.625.
        :param min_coverage: when reaching the last partial utterance, it may or may not have
        enough frames. If at least <min_pad_coverage> of <partials_n_frames> are present,
        then the last partial utterance will be considered by zero-padding the audio. Otherwise,
        it will be discarded. If there aren't enough frames for one partial utterance,
        this parameter is ignored so that the function always returns at least one slice.
        :return: the embedding as a numpy array of float32 of shape (model_embedding_size,). If
        <return_partials> is True, the partial utterances as a numpy array of float32 of shape
        (n_partials, model_embedding_size) and the wav partials as a list of slices will also be
        returned.
        """
        # Compute where to split the utterance into partials and pad the waveform with zeros if
        # the partial utterances cover a larger range.
        wav_slices, mel_slices = self.compute_partial_slices(
            len(wav), rate, min_coverage)
        max_wave_length = wav_slices[-1].stop
        if max_wave_length >= len(wav):
            wav = np.pad(wav, (0, max_wave_length - len(wav)), "constant")

        # Split the utterance into partials and forward them through the model
        mel = audio.wav_to_mel_spectrogram(wav)
        mels = np.array([mel[s] for s in mel_slices])
        with torch.no_grad():
            mels = torch.from_numpy(mels).to(self.device)
            partial_embeds = self(mels)

        # Compute the utterance embedding from the partial embeddings
        raw_embed = torch.mean(partial_embeds, dim=0)
        embed = raw_embed / torch.norm(raw_embed, 2)

        if return_partials:
            return embed, partial_embeds, wav_slices
        return embed

    def embed_speaker(self, wavs: List[np.ndarray], **kwargs):
        """
        Compute the embedding of a collection of wavs (presumably from the same speaker) by
        averaging their embedding and L2-normalizing it.

        :param wavs: list of wavs a numpy arrays of float32.
        :param kwargs: extra arguments to embed_utterance()
        :return: the embedding as a numpy array of float32 of shape (model_embedding_size,).
        """
        raw_embed = np.mean([self.embed_utterance(wav, return_partials=False, **kwargs) for wav in wavs], axis=0)
        return raw_embed / np.linalg.norm(raw_embed, 2)

    def do_gradient_ops(self):
        # Gradient scale
        self.similarity_weight.grad *= 0.01
        self.similarity_bias.grad *= 0.01

        # Gradient clipping
        clip_grad_norm_(self.parameters(), 3, norm_type=2)

    def similarity_matrix(self, embeds):
        """
        Computes the similarity matrix according the section 2.1 of GE2E.

        :param embeds: the embeddings as a tensor of shape (speakers_per_batch,
        utterances_per_speaker, embedding_size)
        :return: the similarity matrix as a tensor of shape (speakers_per_batch,
        utterances_per_speaker, speakers_per_batch)
        """
        speakers_per_batch, utterances_per_speaker = embeds.shape[:2]

        # Inclusive centroids (1 per speaker). Cloning is needed for reverse differentiation
        centroids_incl = torch.mean(embeds, dim=1, keepdim=True)
        centroids_incl = centroids_incl.clone() / torch.norm(centroids_incl, dim=2, keepdim=True)

        # Exclusive centroids (1 per utterance)
        centroids_excl = (torch.sum(embeds, dim=1, keepdim=True) - embeds)
        centroids_excl /= (utterances_per_speaker - 1)
        centroids_excl = centroids_excl.clone() / torch.norm(centroids_excl, dim=2, keepdim=True)

        # Similarity matrix. The cosine similarity of already 2-normed vectors is simply the dot
        # product of these vectors (which is just an element-wise multiplication reduced by a sum).
        # We vectorize the computation for efficiency.
        sim_matrix = torch.zeros(speakers_per_batch, utterances_per_speaker,
                                 speakers_per_batch).to(self.loss_device)
        mask_matrix = 1 - np.eye(speakers_per_batch, dtype=np.int)
        for j in range(speakers_per_batch):
            mask = np.where(mask_matrix[j])[0]
            sim_matrix[mask, :, j] = (embeds[mask] * centroids_incl[j]).sum(dim=2)
            sim_matrix[j, :, j] = (embeds[j] * centroids_excl[j]).sum(dim=1)

        ## Even more vectorized version (slower maybe because of transpose)
        # sim_matrix2 = torch.zeros(speakers_per_batch, speakers_per_batch, utterances_per_speaker
        #                           ).to(self.loss_device)
        # eye = np.eye(speakers_per_batch, dtype=np.int)
        # mask = np.where(1 - eye)
        # sim_matrix2[mask] = (embeds[mask[0]] * centroids_incl[mask[1]]).sum(dim=2)
        # mask = np.where(eye)
        # sim_matrix2[mask] = (embeds * centroids_excl).sum(dim=2)
        # sim_matrix2 = sim_matrix2.transpose(1, 2)

        sim_matrix = sim_matrix * self.similarity_weight + self.similarity_bias
        return sim_matrix

    def loss(self, embeds):
        """
        Computes the softmax loss according the section 2.1 of GE2E.

        :param embeds: the embeddings as a tensor of shape (speakers_per_batch, utterances_per_speaker, embedding_size)
        :return: the loss and the EER for this batch of embeddings.
        """
        speakers_per_batch, utterances_per_speaker = embeds.shape[:2]

        # Loss
        sim_matrix = self.similarity_matrix(embeds)
        sim_matrix = sim_matrix.reshape(
            (speakers_per_batch * utterances_per_speaker, speakers_per_batch))
        ground_truth = np.repeat(np.arange(speakers_per_batch),
                                 utterances_per_speaker)
        target = torch.from_numpy(ground_truth).long().to(self.loss_device)
        loss = self.loss_fn(sim_matrix, target)

        # EER (not backpropagated)
        with torch.no_grad():
            inv_argmax = lambda i: np.eye(
                1, speakers_per_batch, i, dtype=np.int)[0]
            labels = np.array([inv_argmax(i) for i in ground_truth])
            preds = sim_matrix.detach().cpu().numpy()

            # Snippet from https://yangcha.github.io/EER-ROC/
            try:
                fpr, tpr, thresholds = roc_curve(labels.flatten(),
                                                 preds.flatten())
                eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
            except ValueError:
                print(preds.flatten())
                print('preds contain nan:', np.isnan(np.sum(preds.flatten())))
                print(
                    'embeds contain nan:',
                    np.isnan(np.sum(embeds.detach().cpu().numpy().flatten())))
                print('=============')

        return loss, eer


def train(run_id: str, clean_data_root: Path, models_dir: Path,
          umap_every: int, save_every: int, backup_every: int, vis_every: int,
          force_restart: bool, visdom_server: str, no_visdom: bool):
    from resemblyzer.visualizations import Visualizations

    def sync(device: torch.device):
        # For correct profiling (cuda operations are async)
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    # Create a dataset and a dataloader
    dataset = SpeakerVerificationDataset(clean_data_root)
    loader = SpeakerVerificationDataLoader(
        dataset,
        speakers_per_batch,
        utterances_per_speaker,
        num_workers=8,
    )

    # Setup the device on which to run the forward pass and the loss. These can be different,
    # because the forward pass is faster on the GPU whereas the loss is often (depending on your
    # hyperparameters) faster on the CPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # FIXME: currently, the gradient is None if loss_device is cuda
    loss_device = torch.device("cpu")

    # Create the model and the optimizer
    model = VoiceEncoder(device, loss_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate_init)
    init_step = 1

    # Configure file path for the model
    state_fpath = models_dir.joinpath(run_id + ".pt")
    backup_dir = models_dir.joinpath(run_id + "_backups")

    # Load any existing model
    if not force_restart:
        if state_fpath.exists():
            print(
                "Found existing model \"%s\", loading it and resuming training."
                % run_id)
            checkpoint = torch.load(state_fpath)
            init_step = checkpoint["step"]
            model.load_state_dict(checkpoint["model_state"])
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            optimizer.param_groups[0]["lr"] = learning_rate_init
        else:
            print("No model \"%s\" found, starting training from scratch." %
                  run_id)
    else:
        print("Starting the training from scratch.")
    model.train()

    # Initialize the visualization environment
    vis = Visualizations(run_id,
                         vis_every,
                         server=visdom_server,
                         disabled=no_visdom)
    vis.log_dataset(dataset)
    vis.log_params()
    device_name = str(
        torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
    vis.log_implementation({"Device": device_name})

    # Training loop
    profiler = Profiler(summarize_every=10, disabled=False)
    logger = Logger(models_dir)
    for step, speaker_batch in enumerate(loader, init_step):
        profiler.tick("Blocking, waiting for batch (threaded)")

        # Forward pass
        inputs = torch.from_numpy(speaker_batch.data).to(device)
        sync(device)
        profiler.tick("Data to %s" % device)
        embeds = model(inputs)
        sync(device)
        profiler.tick("Forward pass")
        embeds_loss = embeds.view(
            (speakers_per_batch, utterances_per_speaker, -1)).to(loss_device)

        loss, eer = model.loss(embeds_loss)
        sync(loss_device)
        profiler.tick("Loss")
        logger.write_line(f'Step {step},\tLoss {loss},\tEER {eer}')

        # Backward pass
        model.zero_grad()
        loss.backward()
        profiler.tick("Backward pass")
        model.do_gradient_ops()
        optimizer.step()
        profiler.tick("Parameter update")

        # Update visualizations
        # learning_rate = optimizer.param_groups[0]["lr"]
        vis.update(loss.item(), eer, step)

        # Draw projections and save them to the backup folder
        if umap_every != 0 and step % umap_every == 0:
            print("Drawing and saving projections (step %d)" % step)
            backup_dir.mkdir(exist_ok=True)
            projection_fpath = backup_dir.joinpath("%s_umap_%06d.png" %
                                                   (run_id, step))
            embeds = embeds.detach().cpu().numpy()
            vis.draw_projections(embeds, utterances_per_speaker, step,
                                 projection_fpath)
            vis.save()

        # Overwrite the latest version of the model
        if save_every != 0 and step % save_every == 0:
            print("Saving the model (step %d)" % step)
            torch.save(
                {
                    "step": step + 1,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                }, state_fpath)

        # Make a backup
        if backup_every != 0 and step % backup_every == 0:
            print("Making a backup (step %d)" % step)
            backup_dir.mkdir(exist_ok=True)
            backup_fpath = backup_dir.joinpath("%s_bak_%06d.pt" %
                                               (run_id, step))
            torch.save(
                {
                    "step": step + 1,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                }, backup_fpath)

        profiler.tick("Extras (visualizations, saving)")
