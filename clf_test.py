import argparse
import numpy as np
from time import perf_counter as timer
from tqdm import tqdm
from pathlib import Path

from resemblyzer import Predictor


def predict_folder(data_path='../data/clv', exp_dir='exp/combine', num_class=381):
    class_path = Path(exp_dir, 'classes.npy')
    # clf_ckpt_path = Path(exp_dir, 'mlp/mlp_best_val_loss.pt')
    # clf_ckpt_path = Path(exp_dir, 'mlp/mlp_best_val_acc.pt')
    clf_ckpt_path = Path(exp_dir, 'mlp/mlp_e200.pt')
    enc_ckpt_path = 'ckpt/pretrained.pt'
    # enc_ckpt_path = 'ckpt/encoder_mdl_ls_cv_vctk_vc12_bak_1000100.pt'
    predictor = Predictor(clf_ckpt_path, enc_ckpt_path, num_class=num_class, verbose=True)
    classes = np.load(class_path)
    correct = 0
    count = 0
    runtimes = {'pre': [], 'pred': []}
    topk = 2
    try:
        filepaths = list(Path(data_path).glob("**/*.flac"))
        for filepath in tqdm(filepaths):
            label = str(filepath).split('/')[-3]
            start = timer()
            audio = predictor.preprocess(filepath)
            runtimes['pre'].append(timer() - start)
            start = timer()
            top_probs, top_classes = predictor.predict(audio, topk=topk)
            runtimes['pred'].append(timer() - start)
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
    ap.add_argument('-n', '--num_class', type=int, default='384')
    args = ap.parse_args()
    predict_folder(args.data, args.exp, num_class=args.num_class)
