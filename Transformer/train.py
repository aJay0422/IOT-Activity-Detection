import numpy as np
import torch
import torch.nn as nn
import os
from sklearn.linear_model import LogisticRegression

from utils import get_loss_acc, prepare_data, prepare_data_3D
from model import transformer_base, transformer_huge, transformer_large


def train(model, epochs, trainloader, testloader, optimizer, criterion, save_path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Trained on {}".format(device))
    model.to(device)

    # train model
    best_test_acc = 0
    model.train()
    for epoch in range(epochs):
        for X_batch, Y_batch in trainloader:
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)

            # forward
            logits = model(X_batch)
            loss = criterion(logits, Y_batch)
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # evaluate
        with torch.no_grad():
            model.eval()
            # evaluate train
            train_loss, train_acc = get_loss_acc(model, trainloader, criterion)
            # evaluate test
            test_loss, test_acc = get_loss_acc(model, testloader, criterion)

        print("Epoch{}/{}  train loss: {}  test loss: {}  train acc: {}  testacc: {}".format(epoch+1, epochs,
                                                                                             train_loss, test_loss,
                                                                                             train_acc, test_acc))
        # save model weights if  it's the best
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), save_path)
            print("Saved")


def experiment1(size="base"):
    seed = 20220728
    experiment_path = "./experiment"
    if not os.path.exists(experiment_path):
        os.mkdir(experiment_path)

    for i in range(5):
        this_seed = seed + i
        save_path = experiment_path + "/Transformer_{}_{}.pth".format(size, i + 1)
        trainloader, testloader = prepare_data(test_ratio=0.20, seed=this_seed)

        # train model
        if size == "base":
            model = transformer_base()
        elif size == "large":
            model = transformer_large()
        elif size == "huge":
            model = transformer_huge()
        epochs = 200
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
        criterion = nn.CrossEntropyLoss()
        train(model, epochs, trainloader, testloader, optimizer, criterion, save_path)


def experiment2():
    seed = 20220813
    experiment_path = "./experiment_3D_feature"
    if not os.path.exists(experiment_path):
        os.mkdir(experiment_path)

    for i in range(5):
        this_seed = seed + i
        save_path = experiment_path + "/Transformer_huge_{}.pth".format(i + 1)
        trainloader, testloader = prepare_data_3D(test_ratio=0.2, seed=this_seed, scale=True)

        # train model
        model = transformer_huge(n_features=51)
        epochs = 200
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
        criterion = nn.CrossEntropyLoss()
        train(model, epochs, trainloader, testloader, optimizer, criterion, save_path)


def experiment3():
    """
    Does shuffling affect the accuracy?
    """
    seed = 20220728
    experiment_path = "./experiment_shuffle"
    if not os.path.exists(experiment_path):
        os.mkdir(experiment_path)

    # no shuffle
    setting_path = experiment_path + "/no_shuffle"
    if not os.path.exists(setting_path):
        os.mkdir(setting_path)
    for i in range(5):
        this_seed = seed + i
        save_path = setting_path + f"/Transformer_huge_no_shuffle_{i+1}.pth"
        trainloader, testloader = prepare_data(seed=this_seed, shuffle_frame=False)
        model = transformer_huge()
        epochs = 200
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
        criterion = nn.CrossEntropyLoss()
        train(model, epochs, trainloader, testloader, optimizer, criterion, save_path)

    # shuffle
    setting_path = experiment_path + "/shuffle"
    if not os.path.exists(setting_path):
        os.mkdir(setting_path)
    for shuffle_id in range(3):
        for i in range(5):
            this_seed = seed + i
            save_path = setting_path + f"/Transformer_huge_shuffle{shuffle_id+1}_{i+1}.pth"
            trainloader_shuffle, testloader_shuffle = prepare_data(seed=this_seed, shuffle_frame=True)
            model = transformer_huge()
            epochs = 200
            optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
            criterion = nn.CrossEntropyLoss()
            train(model, epochs, trainloader_shuffle, testloader_shuffle, optimizer, criterion, save_path)


def experiment4(index=1):
    """extract features using Transformer and fit a Logistic Regression"""
    # Prepare model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = transformer_huge().to(device)
    net.load_state_dict(torch.load(f"experiment/transformer_huge_{index}.pth", map_location=device))

    # Prepare dataset
    trainloader, testloader = prepare_data(seed=20220728+index-1)

    feature_train = []
    Y_train = []
    feature_test = []
    Y_test = []
    with torch.no_grad():
        net.eval()
        for X_batch, Y_batch in trainloader:
            features = net.forward_features(X_batch.to(device))
            feature_train.append(features)
            Y_train.append(Y_batch)

        for X_batch, Y_batch in testloader:
            features = net.forward_features(X_batch.to(device))
            feature_test.append(features)
            Y_test.append(Y_batch)

    feature_train = torch.cat(feature_train, dim=0).cpu().numpy()
    Y_train = torch.cat(Y_train, dim=0).numpy()
    feature_test = torch.cat(feature_test, dim=0).cpu().numpy()
    Y_test = torch.cat(Y_test, dim=0).numpy()

    clf = LogisticRegression(max_iter=1000)
    clf.fit(feature_train, Y_train)
    Y_test_pred = clf.predict(feature_test)
    acc_test = np.mean(Y_test_pred == Y_test)
    print(acc_test)


if __name__ == "__main__":
    experiment4(1)
    experiment4(2)
    experiment4(3)
    experiment4(4)
    experiment4(5)
