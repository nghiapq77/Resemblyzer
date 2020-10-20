from time import perf_counter as timer
import torch

from resemblyzer import VoiceEncoder, preprocess_wav
from resemblyzer.classifier import MLP


class Predictor():
    def __init__(self,
                 clf_ckpt_path='exp/clv/mlp/mlp_best_val_loss.pt',
                 enc_ckpt_path='ckpt/pretrained.pt',
                 device=torch.device('cuda'),
                 num_class=381,
                 verbose=False):
        start = timer()
        self.encoder = VoiceEncoder(device=device, loss_device=device)
        self.encoder.load_ckpt(enc_ckpt_path, device)
        self.encoder.eval()
        self.classifier = MLP(num_class=num_class)
        self.classifier.load_ckpt(clf_ckpt_path, device)
        self.classifier.eval()
        if verbose:
            print(
                f'Encoder and classifier models loaded successfully in {timer() - start}s'
            )

    def preprocess(self, f):
        """
        Applies preprocessing operations to a waveform either on disk or in memory such that
        The waveform will be resampled to match the data hyperparameters.

        :param f: either a filepath to an audio file or the waveform as a numpy array of floats.
        """
        return preprocess_wav(f)

    def predict(self, audio, topk=2):
        """
        Predict top_k classes with highest probabilities.

        :param audio: preprocessed waveform.
        :param topk: Keep topk classes with highest probabilities.
        """
        embed = self.encoder.embed_utterance(audio)
        inp = embed.unsqueeze(dim=0)
        top_probs, top_classes = self.classifier.predict(inp, topk=topk)
        return top_probs, top_classes
