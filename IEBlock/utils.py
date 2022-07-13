import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


FEATURE_ARCHIVE = "../feature_archive/"
LABEL_ENCODER = {'no_interaction': 0,
                 'open_close_fridge': 1,
                 'put_back_item': 2,
                 'screen_interaction': 3,
                 'take_out_item': 4}


class mydataset(Dataset):
    def __init__(self, X, Y):
        self.Data = torch.Tensor(X)
        self.Label = torch.LongTensor(Y)

    def __getitem__(self, item):
        return self.Data[item], self.Label[item]

    def __len__(self):
        return len(self.Label)


def prepare_data(test_ratio, seed=20220707):
    all_data = np.load(FEATURE_ARCHIVE + "all_feature_interp971.npz")
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
    trainloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    testloader = DataLoader(test_dataset, batch_size=20)

    return trainloader, testloader


def prepare_encoded_data(encoder, trainloader, testloader):
    # encode train data
    X_train_encoded = []
    Y_train = []
    for X_batch, Y_batch in trainloader:
        X_train_encoded.append(encoder(X_batch))
        Y_train.append(Y_batch)
    X_train_encoded = torch.cat(X_train_encoded, dim=0)
    Y_train = torch.cat(Y_train, dim=0)
    train_dataset = mydataset(X_train_encoded, Y_train)
    trainloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # encode test data
    X_test_encoded = []
    Y_test = []
    for X_batch, Y_batch in testloader:
        X_test_encoded.append(encoder(X_batch))
        Y_test.append(Y_batch)
    X_test_encoded = torch.cat(X_test_encoded, dim=0)
    Y_test = torch.cat(Y_test, dim=0)
    test_dataset = mydataset(X_test_encoded, Y_test)
    testloader = DataLoader(test_dataset, batch_size=20)

    return trainloader, testloader


def get_loss_acc(model, dataloader, criterion):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    correct = 0
    total = 0
    total_loss = 0
    num_batches = 0
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


def trainAE(model, epochs, trainloader, testloader, optimizer, criterion):
    """train an autoencoder"""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # train model
    min_test_loss = 1e10
    model.train()
    for epoch in range(epochs):
        train_loss = 0
        num_batches = 0
        for X_batch, Y_batch in trainloader:
            num_batches += 1
            X_batch = X_batch.to(device)
            # forward
            outputs = model(X_batch)
            loss = criterion(outputs, X_batch)
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
        train_loss /= num_batches

        # evaluate
        test_loss = 0
        with torch.no_grad():
            model.eval()
            num_batches = 0
            for X_batch, Y_batch in testloader:
                X_batch = X_batch.to(device)
                num_batches += 1
                outputs = model(X_batch)
                loss = criterion(outputs, X_batch)
                test_loss += loss.item()
        test_loss /= num_batches

        # save model weights if it's the best
        if test_loss < min_test_loss:
            min_test_loss = test_loss
            torch.save({"model_state_dict": model.state_dict()}, "Best_AE.pt")

        print("Epoch: {}/{}  train loss: {}  test loss: {}".format(epoch + 1, epochs, train_loss, test_loss))

    return model


def trainMLP(model, epochs, trainloader, testloader, optimizer, criterion, model_name):
    """train a mlp classifier"""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # train model
    best_test_acc = 0
    model.train()
    for epoch in range(epochs):
        train_loss = 0
        num_batches = 0
        for X_batch, Y_batch in trainloader:
            num_batches += 1
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
            # forward
            outputs = model(X_batch)
            loss = criterion(outputs, Y_batch)
            # backward
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

            train_loss += loss.item()
        train_loss /= num_batches

        # evaluate
        with torch.no_grad():
            model.eval()
            # evaluate train
            train_loss, train_acc = get_loss_acc(model, trainloader, nn.CrossEntropyLoss())
            # evaluate test
            test_loss, test_acc = get_loss_acc(model, testloader, nn.CrossEntropyLoss())

        print("Epoch: {}/{}  train loss: {}  train acc: {}  test loss: {}  test acc: {}".format(epoch + 1, epochs,
                                                                                                train_loss, train_acc,
                                                                                                test_loss, test_acc))

        # save model weights if it's the best
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save({"model_state_dict": model.state_dict()}, "{}.pt".format(model_name))
            print("Saved")

    return model
