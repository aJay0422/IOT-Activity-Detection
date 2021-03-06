import matplotlib.cm
import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix

from torch.utils.data import Dataset, DataLoader


FEATURE_ARCHIVE = "../feature_archive/"
LABEL_ENCODER = {'no_interaction': 0,
                 'open_close_fridge': 1,
                 'put_back_item': 2,
                 'screen_interaction': 3,
                 'take_out_item': 4}
LABEL_DECODER = {0:"no_interaction",
                 1:"open_close_fridge",
                 2:"put_back_item",
                 3:"screen_interaction",
                 4:"take_out_item"}


class mydataset(Dataset):
    def __init__(self, X, Y):
        self.Data = torch.Tensor(X)
        self.Label = torch.LongTensor(Y)

    def __getitem__(self, item):
        return self.Data[item], self.Label[item]

    def __len__(self):
        return len(self.Label)


def prepare_data(test_ratio, seed=20220712):
    all_data = np.load(FEATURE_ARCHIVE + "all_feature_interp951.npz")
    X_all = all_data["X"]
    Y_all = all_data["Y"]

    # reshape data to (n_samples, n_features, T_k)
    X_all = X_all.reshape(-1, 100, 34).transpose(0, 2, 1)

    # set random seed
    np.random.seed(seed)

    # split data
    n_samples = X_all.shape[0]
    test_size = int(n_samples * test_ratio)
    perm = np.random.permutation(n_samples)
    test_idx = perm[: test_size]
    train_idx = perm[test_size:]
    X_train = X_all[train_idx]
    Y_train = Y_all[train_idx]
    X_test = X_all[test_idx]
    Y_test = Y_all[test_idx]

    # create dataset and dataloader
    train_dataset = mydataset(X_train, Y_train)
    test_dataset = mydataset(X_test, Y_test)
    trainloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    testloader = DataLoader(test_dataset, batch_size=20)

    return trainloader, testloader


def get_loss_acc(model, dataloader, criterion):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    correct = 0
    total = 0
    total_loss = 0
    num_batches = 0
    with torch.no_grad():
        model.eval()
        for X_batch, Y_batch in dataloader:
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
            total += len(Y_batch)
            num_batches += 1
            outputs = model(X_batch)
            y_pred = torch.argmax(outputs, dim=1)
            correct += torch.sum(y_pred == Y_batch).cpu().numpy()
            loss = criterion(outputs, Y_batch)
            total_loss += loss.item()
    acc = correct / total
    total_loss = total_loss / num_batches

    return total_loss, acc


def count_parameters(model, check_trainable=False):
    if check_trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


def draw_confusion_matrix(y_pred, y_true, title="Transformer Huge", save=False):
    plt.style.use("ggplot")
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"
    plt.rcParams["axes.titleweight"] = "bold"
    fig, ax = plt.subplots(1, 1, figsize=(3, 3), dpi=320)
    conf_mx = confusion_matrix(y_true, y_pred)
    ax.matshow(conf_mx, cmap=matplotlib.cm.get_cmap("Blues"))

    n_classes = len(LABEL_ENCODER)
    for i in range(n_classes):
        for j in range(n_classes):
            text = ax.text(j, i, conf_mx[i, j],
                           ha="center",
                           va="center",
                           color="black")
    ax.grid(False)

    plt.xticks(np.arange(n_classes), [cls for cls in LABEL_ENCODER.keys()],
               fontsize=6, rotation=45)
    plt.yticks(np.arange(n_classes), [cls for cls in LABEL_ENCODER.keys()],
               fontsize=6)
    ax.xaxis.set_ticks_position("bottom")
    plt.title("Confusion Matrix: {}".format(title), fontsize=10)
    plt.show()

    if save:
        fig.savefig("Confusion Matrix {}.png".format(title), format="png")
