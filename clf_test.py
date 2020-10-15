import numpy as np
import os
import torch
from time import time
from tqdm import tqdm
import sys
from resemblyzer import VoiceEncoder, preprocess_wav
from resemblyzer.classifier import MLP

device = torch.device('cuda')
def normalize(x):
    """
    Normalize to 0-1
    """
    return (x - np.min(x)) / (np.max(x) - np.min(x))


def load_models(exp_dir='exp/clv', num_class=381):
    encoder = VoiceEncoder(device=device, loss_device=device, ckpt_path='ckpt/pretrained.pt')
    classifier = MLP(projection_size=num_class)
    classifier.load_state_dict(torch.load(os.path.join(exp_dir, 'mlp/mlp_best_val_loss.pkl')))
    classifier.to(device)
    classifier.eval()
    return encoder, classifier


def predict_folder(data_path, num_class=381):
    fmt = '.flac'
    data_name = data_path.split('/')[-1]
    exp_dir = os.path.join('exp', data_name)
    class_path = os.path.join(exp_dir, 'classes.npy')
    encoder, classifier = load_models(exp_dir, num_class=num_class)
    classes = np.load(class_path)
    correct = 0
    count = 0
    runtimes = []
    try:
        for (dirpath, dirnames, filenames) in tqdm(os.walk(data_path)):
            for filename in filenames:
                if filename.endswith(tuple(fmt)):
                    filepath = dirpath + '/' + filename
                    label = filepath.split('/')[-3]

                    start = time()
                    audio = preprocess_wav(filepath)
                    # print(f'Preprocess time: {time() - start}')
                    embed = encoder.embed_utterance(audio)
                    inp = embed.unsqueeze(dim=0)
                    out = classifier(inp)
                    pred = out.max(dim=1)[1].cpu().numpy()
                    runtimes.append(time() - start)
                    # out = out.detach().cpu().numpy().squeeze()
                    # print(pred, normalize(out)[pred])
                    count += 1
                    if classes[pred][0] == label:
                        correct += 1

                    # if count == 2000:
                        # print(f'Acc: {correct / count}')
                        # print(f'Avg runtime: {np.mean(runtimes)}')
                        # sys.exit(1)
        print(f'Acc: {correct / count}')
        print(f'Avg runtime: {np.mean(runtimes)}')
    except KeyboardInterrupt:
        print(f'Acc: {correct / count}')
        print(f'Avg runtime: {np.mean(runtimes)}')


if __name__ == '__main__':
    predict_folder('../data/combine', num_class=381)
