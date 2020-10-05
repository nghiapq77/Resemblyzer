import numpy as np
import pickle
from pathlib import Path
from itertools import groupby
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import datasets

from resemblyzer import preprocess_wav, VoiceEncoder

### data
# X = np.load('./embeds_from_pre.npy', allow_pickle=True)
# y = np.load('./labels_from_pre.npy', allow_pickle=True)
# print(len(X))
encoder = VoiceEncoder(ckpt_path='ckpt/pretrained.pt')
wav_fpaths = list(Path("/home/ubuntu/speaker-recognition/data/clv").glob("**/*.wav"))

# Group the wavs per speaker and load them using the preprocessing function provided with
# resemblyzer to load wavs in memory. It normalizes the volume, trims long silences and resamples
# the wav to the correct sampling rate.
speaker_wavs = {
    speaker: list(map(preprocess_wav, wav_fpaths))
    for speaker, wav_fpaths in groupby(
        tqdm(wav_fpaths, "Preprocessing wavs", len(wav_fpaths), unit="wavs"),
        lambda wav_fpath: wav_fpath.parent.stem)
}
X = []
for wavs in speaker_wavs.values():
    for wav in wavs:
        X.append(encoder.embed_utterance(wav))
X = np.array(X)
y = list(map(lambda wav_fpath: wav_fpath.parent.stem, wav_fpaths))

# Creating training and test split
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.1,
                                                    random_state=42,
                                                    stratify=y)

# Feature Scaling
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

# Training a SVM classifier using SVC class
svm = SVC(kernel='linear', random_state=1, C=0.1, probability=False)
svm.fit(X_train_std, y_train)

# rfc = RandomForestClassifier(random_state=42)
# rfc.fit(X_train_std, y_train)

# Model performance
y_pred = svm.predict(X_train_std)
# y_ = svm.decision_function(X_train_std)
# y_prob = svm.predict_proba(X_train_std)
# print(y_prob[0])
print('Accuracy train: %.3f' % accuracy_score(y_train, y_pred))
y_pred = svm.predict(X_test_std)
print('Accuracy test: %.3f' % accuracy_score(y_test, y_pred))

# Save to file in the current working directory
pkl_filename = "ckpt/svm.pkl"
with open(pkl_filename, 'wb') as f:
    pickle.dump(svm, f)
