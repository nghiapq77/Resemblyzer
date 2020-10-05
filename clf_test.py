import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


### Model loaded
pkl_filename = 'ckpt/svm.pkl'
with open(pkl_filename, 'rb') as file:
    model = pickle.load(file)

### data
X = np.load('./data/embeds/embeds_from_pre.npy', allow_pickle=True)
y = np.load('./data/embeds/labels_from_pre.npy', allow_pickle=True)
print(len(X))

### Creating training and test split
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.1,
                                                    random_state=42,
                                                    stratify=y)

### Feature Scaling
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

### Model performance
y_pred = model.predict(X_train_std)
print('Accuracy train: %.3f' % accuracy_score(y_train, y_pred))
y_pred = model.predict(X_test_std)
print('Accuracy test: %.3f' % accuracy_score(y_test, y_pred))
