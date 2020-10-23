import argparse
from multiprocess.pool import ThreadPool
from tqdm import tqdm
from pathlib import Path

from resemblyzer import Augmenter, sampling_rate
from resemblyzer.classifier import load_data


def prepare(args):
    data_path = args.data
    ckpt_path = args.embedding_ckpt
    outn = data_path.split('/')[-1]
    if outn == '':
        outn = data_path.split('/')[-2]
    save_path = Path('exp', outn)
    if not save_path.exists():
        save_path.mkdir()
    load_data(ckpt_path=ckpt_path, data_path=data_path, save_path=save_path)


def augment(args):
    aug = Augmenter(sampling_rate=sampling_rate)
    data_path = args.data
    inn = data_path.split('/')[-1]
    if inn == '':
        inn = data_path.split('/')[-2]
    outn = inn + '_aug'
    data_path = Path(data_path)
    filepaths = list(data_path.glob("**/*.flac"))

    def augment_file(filepath):
        filename = filepath.stem
        save_filename = str(filename) + '_aug'
        save_orig_path = str(filepath).replace(inn, outn)
        save_filepath = save_orig_path.replace(filename, save_filename)
        save_filepath = Path(save_filepath)
        save_filepath.parent.mkdir(parents=True, exist_ok=True)
        data, sr = aug.load(filepath)
        data = aug.resample(data, sr)
        augmented = aug.augment(data)
        aug.save(save_filepath, augmented)
        if args.add:
            aug.save(save_orig_path, data)

    # Multi-threading
    with ThreadPool(8) as pool:
        list(tqdm(pool.imap(augment_file, filepaths), 'Aug', len(filepaths), unit="files"))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('function', type=str)
    ap.add_argument('-a', '--add', action='store_true')
    ap.add_argument('-d', '--data', default='../data/clv')
    ap.add_argument('-e', '--exp', default='exp/combine_500')
    ap.add_argument('-n', '--num_class', type=int, default='498')
    ap.add_argument('-ckpt', '--embedding_ckpt', default='ckpt/pretrained.pt')
    args = ap.parse_args()
    globals()[args.function](args)
