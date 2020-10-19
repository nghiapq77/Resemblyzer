import argparse
import numpy as np
from time import perf_counter as timer
from tqdm import tqdm
from pathlib import Path
import sys
import torch
from torch.nn.functional import softmax

from resemblyzer import VoiceEncoder, preprocess_wav
from resemblyzer.classifier import MLP


def normalize(x):
    """
    Normalize to 0-1
    """
    return (x - np.min(x)) / (np.max(x) - np.min(x))


def load_models(clf_ckpt_path='exp/clv/mlp/mlp_best_val_loss.pt',
                enc_ckpt_path='ckpt/pretrained.pt',
                device=torch.device('cuda'),
                num_class=381,
                verbose=False):
    start = timer()
    encoder = VoiceEncoder(device=device, loss_device=device)
    encoder_ckpt = torch.load(enc_ckpt_path, map_location="cpu")
    encoder.load_state_dict(encoder_ckpt["model_state"], strict=False)
    encoder.to(device)
    encoder.eval()
    classifier = MLP(num_class=num_class)
    classifier.load_state_dict(torch.load(clf_ckpt_path))
    classifier.to(device)
    classifier.eval()
    if verbose:
        print(f'Encoder and classifier models loaded successfully in {timer() - start}s')
    return encoder, classifier


def predict(audio, encoder, classifier, runtimes=None, topk=2):
    # TODO: seperate time calculation
    start = timer()
    embed = encoder.embed_utterance(audio)
    runtimes['emb'].append(timer() - start)
    inp = embed.unsqueeze(dim=0)
    start = timer()
    out = classifier(inp)
    runtimes['clf'].append(timer() - start)
    prob = softmax(out, dim=1)
    top_probs, top_classes = prob.topk(topk, dim=1)
    return top_probs.detach().cpu().numpy()[0], top_classes.cpu().numpy()[0]


def predict_folder(data_path='../data/clv', exp_dir='exp/combine', num_class=381):
    class_path = Path(exp_dir, 'classes.npy')
    clf_ckpt_path = Path(exp_dir, 'mlp/mlp_best_val_loss.pt')
    encoder, classifier = load_models(clf_ckpt_path=clf_ckpt_path, enc_ckpt_path='ckpt/pretrained.pt', num_class=num_class, verbose=True)
    classes = np.load(class_path)
    correct = 0
    count = 0
    runtimes = {'pre': [], 'emb': [], 'clf': []}
    topk = 2
    try:
        filepaths = list(Path(data_path).glob("**/*.flac"))
        for filepath in tqdm(filepaths):
            label = str(filepath).split('/')[-3]
            start = timer()
            audio = preprocess_wav(filepath)
            runtimes['pre'].append(timer() - start)
            top_probs, top_classes = predict(audio, encoder, classifier, runtimes, topk=topk)
            pred = classes[top_classes[0]]
            count += 1
            if pred == label:
                correct += 1
            elif top_probs[0] < 0.7:
                tqdm.write(f'{filepath}')
                for i in range(topk):
                    tqdm.write(str(classes[top_classes[i]]) + ' ' + str(top_probs[i]))
                tqdm.write('==============')
            else:
                tqdm.write(f'{filepath}')
                for i in range(topk):
                    tqdm.write(str(classes[top_classes[i]]) + ' ' + str(top_probs[i]))
                tqdm.write('==============')
        print(f'Acc: {correct / count}')
        for key in runtimes.keys():
            print(f'Avg {key} runtime: {np.mean(runtimes[key])}')
    except KeyboardInterrupt:
        print(f'Acc: {correct / count}')
        for key in runtimes.keys():
            print(f'Avg {key} runtime: {np.mean(runtimes[key])}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-d', '--data', default='../data/clv')
    ap.add_argument('-e', '--exp', default='exp/combine')
    args = ap.parse_args()
    predict_folder(args.data, args.exp, num_class=381)
