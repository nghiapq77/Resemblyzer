import numpy as np
import pickle
import os
from pathlib import Path
from datetime import datetime
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from scipy.special import softmax
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from resemblyzer.hparams import model_embedding_size


def load_data(from_path=None, ckpt_path=None, data_path=None, save_path=None):
    if from_path is None:
        if ckpt_path is None:
            raise Exception('No checkpoint path provided')

        from resemblyzer import preprocess_wav, VoiceEncoder
        from tqdm import tqdm

        device = torch.device('cuda')
        encoder = VoiceEncoder(device=device, loss_device=device)
        encoder.load_ckpt(ckpt_path, device=device)
        encoder.eval()
        wav_fpaths = list(Path(data_path).glob("**/*.flac"))

        # Preprocess and save encoded utterance and label to list
        X = []
        y = []
        for wav_fpath in tqdm(wav_fpaths):
            wav = preprocess_wav(wav_fpath)
            X.append(encoder.embed_utterance(wav).cpu().numpy())
            y.append(wav_fpath.parent.parent.stem)

        # Save for testing
        if save_path is not None:
            np.save(Path(save_path, 'embeds.npy'), X)
            np.save(Path(save_path, 'labels.npy'), y)
        else:
            raise Exception('No save_path provided')
    else:
        X = np.load(Path(from_path, 'embeds.npy'), allow_pickle=True)
        y = np.load(Path(from_path, 'labels.npy'), allow_pickle=True)
    return X, y


def split_data(X, y, test_size=0.2):
    # Creating training and test split
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=test_size,
                                                        random_state=42,
                                                        stratify=y)
    return X_train, X_test, y_train, y_test


def scale_data(X_train, X_test=None, method='standard'):
    # Feature Scaling
    if method == 'standard':
        sc = StandardScaler()
    elif method == 'minmax':
        sc = MinMaxScaler()
    else:
        raise Exception('No such scale method defined')

    sc.fit(X_train)
    X_train_std = sc.transform(X_train)
    if X_test is None:
        X_test_std = None
    else:
        X_test_std = sc.transform(X_test)
    return X_train_std, X_test_std


class Logger:
    """
    Logger for classifier
    """
    def __init__(self, root):
        self.text_file = open(Path(root, "log.txt"), "w")

        start_time = str(datetime.now().strftime("%A %d %B %Y at %H:%M"))
        self.write_line(f"Creating log on {start_time}")
        self.write_line("=".center(100, '='))

    def write_line(self, line):
        self.text_file.write("%s\n" % line)

    def finalize(self):
        self.write_line("=".center(100, '='))
        end_time = str(datetime.now().strftime("%A %d %B %Y at %H:%M"))
        self.write_line(f"Finished on {end_time}")
        self.text_file.close()


class SVM():
    def __init__(self, kernel='linear', C=0.1, degree=3):
        self.model = SVC(kernel=kernel, random_state=1, C=C, degree=degree, probability=False)


def train_svm(from_path=None, ckpt_path=None, data_path=None, save_path=None):
    svm = SVM(kernel='linear', C=1)

    ### data
    X, y = load_data(from_path=from_path, ckpt_path=ckpt_path, data_path=data_path, save_path=save_path)

    # Creating training and test split
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Feature Scaling
    X_train_std, X_test_std = scale_data(X_train, X_test)
    # Training a SVM classifier using SVC class
    svm.model.fit(X_train, y_train)

    # Model performance
    y_pred = svm.model.predict(X_train)
    p = np.array(svm.model.decision_function(X_train))  # decision is a voting function
    prob = softmax(p)
    print(p[0], prob[0], y_pred[0], y_train[0])
    print('Accuracy train: %.3f' % accuracy_score(y_train, y_pred))
    y_pred = svm.model.predict(X_test_std)
    print('Accuracy test: %.3f' % accuracy_score(y_test, y_pred))

    # Save to file in the current working directory
    pkl_filename = "exp/clv/svm.pkl"
    with open(pkl_filename, 'wb') as f:
        pickle.dump(svm.model, f)


def train_mul_svm(from_path='exp/combine', scale=False, test_size=0.2):
    kernels = ['rbf', 'poly', 'linear']

    ### data
    X, y = load_data(from_path=from_path)

    # Creating training and test split
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=test_size)

    # Feature Scaling
    if scale:
        X_train, X_test = scale_data(X_train, X_test)

    for i, kernel in enumerate(kernels):
        svm = SVM(kernel=kernel, C=1)
        svm.model.fit(X_train, y_train)
        # Model performance
        y_pred = svm.model.predict(X_train)
        print(f'{kernel} - Accuracy train: {accuracy_score(y_train, y_pred)}')
        y_pred = svm.model.predict(X_test)
        print(f'{kernel} - Accuracy test: {accuracy_score(y_test, y_pred)}')


class EmbedDataset(Dataset):
    def __init__(self, from_path='./exp/clv', is_train=True, test_ratio=0.2):
        embeds, labels = load_data(from_path)
        # embeds, _ = scale_data(embeds)
        le = LabelEncoder()
        # self.embeds = torch.from_numpy(embeds)
        self.embeds = embeds
        self.labels = le.fit_transform(labels)

        # Save classes
        np.save(os.path.join(from_path, 'classes.npy'), le.classes_)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        embed = self.embeds[idx]
        label = self.labels[idx]
        return embed, label


class ConvNet(nn.Module):
    def __init__(self, n_class=18, in_channels=1, ckpt_path=None):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, 64, 3, stride=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(3)
        self.conv2 = nn.Conv1d(64, 64, 3)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(3)
        self.avgPool = nn.AvgPool1d(27)
        self.fc1 = nn.Linear(64, n_class)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)

        x = self.avgPool(x)
        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        return x


class MLP(nn.Module):
    def __init__(self, inp_dim=256, num_class=269, hidden_size=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(inp_dim, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, num_class)
        )

    def forward(self, x):
        return self.net(x)

    def load_ckpt(self,
                  ckpt_path='exp/clv/mlp/mlp_best_val_loss.pt',
                  device=torch.device('cuda')):
        self.load_state_dict(torch.load(ckpt_path))
        self.to(device)

    def predict(self, inp, topk=2):
        out = self.net(inp)
        prob = F.softmax(out, dim=1)
        top_probs, top_classes = prob.topk(topk, dim=1)
        return top_probs.detach().cpu().numpy()[0], top_classes.cpu().numpy()[0]


def train_mlp(args):
    # Specify device
    device = torch.device(args.device)

    # Model saved path
    save_path = os.path.join(args.data_path, 'mlp')
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    # Logger
    logger = Logger(save_path)

    # Create a dataset and a dataloader
    dataset = EmbedDataset(from_path=args.data_path)
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(args.val_split * dataset_size))
    np.random.seed(42)
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    train_loader = DataLoader(dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=8, pin_memory=True)
    val_loader = DataLoader(dataset, batch_size=args.batch_size, sampler=val_sampler, num_workers=8, pin_memory=True)

    # Create the model and the optimizer
    model = MLP(inp_dim=model_embedding_size, num_class=args.num_class).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.1)

    best_val_loss = 9999
    epoch_best_val_loss = 0
    best_val_acc = 0
    epoch_best_val_acc = 0
    # Training
    for epoch in range(args.epochs + 1):
        model.train()
        tot_loss = []
        correct = 0
        for data, target in train_loader:
            # Data to device
            inputs = data.to(device)
            labels = target.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Calculate loss and accuracy
            tot_loss.append(loss.detach().cpu().numpy())
            preds = outputs.max(dim=1)[1]
            correct += preds.eq(labels).cpu().numpy().sum()

            # Backward pass
            loss.backward()
            optimizer.step()
        scheduler.step()
        print(f'Epoch {epoch}, TrainLoss: {np.mean(tot_loss)}, Acc: {correct / len(train_indices)}')
        logger.write_line(f'Epoch {epoch}, TrainLoss: {np.mean(tot_loss)}, Acc: {correct / len(train_indices)}')

        if epoch % args.val_epoch == 0:
            model.eval()
            val_losses = []
            correct = 0
            for data, target in val_loader:
                # Data to device
                inputs = data.to(device)
                labels = target.to(device)

                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                # Calculate loss and accuracy
                val_losses.append(loss.detach().cpu().numpy())
                preds = outputs.max(dim=1)[1]
                correct += preds.eq(labels).cpu().numpy().sum()

            val_loss = np.mean(val_losses)
            val_acc = correct / len(val_indices)
            print(f'Epoch {epoch}, ValLoss: {val_loss}, Acc: {val_acc}')
            logger.write_line(f'Epoch {epoch}, ValLoss: {val_loss}, Acc: {val_acc}')

            # Saving
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epoch_best_val_loss = epoch
                torch.save(model.state_dict(), os.path.join(save_path, 'mlp_best_val_loss.pt'))
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                epoch_best_val_acc = epoch
                torch.save(model.state_dict(), os.path.join(save_path, 'mlp_best_val_acc.pt'))
    print(f'Best val loss {best_val_loss} at epoch {epoch_best_val_loss}')
    logger.write_line(f'Best val loss {best_val_loss} at epoch {epoch_best_val_loss}')
    print(f'Best val acc {best_val_acc} at epoch {epoch_best_val_acc}')
    logger.write_line(f'Best val acc {best_val_acc} at epoch {epoch_best_val_acc}')
    logger.finalize()


if __name__ == '__main__':
    pass
