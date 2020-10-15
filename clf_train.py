import argparse
import os
from resemblyzer.classifier import load_data, train_mul_svm, train_mlp


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-t', '--type', default='mlp')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('-d', '--data_path', default='exp/combine')
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--val_epoch', type=int, default=5)
    ap.add_argument('--lr', type=float, default=0.001)
    ap.add_argument('--val_split', type=float, default=0.2)
    ap.add_argument('--epochs', type=int, default=200)
    ap.add_argument('-n', '--num_class', type=int, default=381)
    ap.add_argument('-e', '--embedding_ckpt', default='ckpt/pretrained.pt')
    args = ap.parse_args()
    if args.type == 'mlp':
        train_mlp(args)
    elif args.type == 'svm':
        # train_SVM(from_path='exp/clv', ckpt_path='exp/clv/clv_trans.pt', data_path='data/librispeech_train-clean-100')
        # train_SVM(ckpt_path='ckpt/pretrained.pt', data_path='../data/LibriSpeech/train-clean-100')
        # train_SVM(ckpt_path='ckpt/pretrained.pt', data_path='../data/clv_new1', save_path='exp/combine')
        train_mul_svm(from_path=args.data_path, scale=False)
    elif args.type == 'prepare':
        outn = args.data_path.split('/')[-1]
        if outn == '':
            outn = args.data_path.split('/')[-2]
        save_path = os.path.join('exp', outn)
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        load_data(ckpt_path=args.embedding_ckpt, data_path=args.data_path, save_path=save_path)
