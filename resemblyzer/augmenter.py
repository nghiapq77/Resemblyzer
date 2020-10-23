import numpy as np
import nlpaug.augmenter.audio as naa
import librosa
import soundfile as sf
from copy import deepcopy

from resemblyzer.hparams import sampling_rate


class Augmenter:
    def __init__(self, sampling_rate=16000):
        self.methods = [
            naa.CropAug(),
            naa.LoudnessAug(),
            naa.NoiseAug(color='white'),
            naa.VtlpAug(sampling_rate=sampling_rate),
            naa.MaskAug(sampling_rate=sampling_rate, coverage=0.1, mask_with_noise=False),

            # naa.ShiftAug(sampling_rate=sampling_rate, duration=1),
            # naa.SpeedAug(factor=(0.9, 1.1)),
        ]
        self.sampling_rate = sampling_rate

    def load(self, fpath):
        data, sr = librosa.load(fpath)
        return data, sr

    def save(self, fpath, data):
        sf.write(fpath, data, self.sampling_rate, format='flac')

    def resample(self, data, sr):
        return librosa.resample(data, sr, self.sampling_rate, res_type='fft')

    def augment(self, data):
        augmented = deepcopy(data)
        num_methods = np.random.randint(1, len(self.methods))
        # num_methods = 1
        for aug in np.random.choice(self.methods, num_methods, replace=False):
            augmented = aug.augment(augmented)
        return augmented


if __name__ == '__main__':
    fpath = 'data/clv/BaoHuy/1/BaoHuy-1-File01.flac'
    data, sr = librosa.load(fpath)
    resampled = librosa.resample(data, sr, sampling_rate, res_type='fft')
    a = Augmenter()
    augmented = a.augment(resampled)
    sf.write('a.flac', augmented, sampling_rate, format='flac')
