"""
Train different classifiers with weighted samples
"""
import os.path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from Transformer.model import transformer_huge
from Transformer.utils import prepare_data, get_loss_acc
from confidence_interpolation import extract_scores


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def save_frame_attention():
    net = transformer_huge().to(device)
    net.load_state_dict(torch.load("Transformer/experiment/Transformer_huge_1.pth", map_location=device))

    file = np.load("feature_archive/all_feature_interp951.npz", allow_pickle=True)
    X = file["X"].reshape(-1, 100, 34).transpose(0, 2, 1)
    Y = file["Y"]

    attention_by_frame = np.zeros((len(Y), 100))

    with torch.no_grad():
        net.eval()
        for i in range(len(Y)):
            video = torch.Tensor(X[i]).unsqueeze(0).to(device)
            logit = net(video)
            attention = net.blocks[6].attn.attn[0, :, 0, 1:].cpu().numpy()
            attention = np.mean(attention, axis=0)
            attention = (attention - attention.min()) / (attention.max() - attention.min())
            attention_by_frame[i, :] = attention

    print(attention_by_frame.shape)
    np.save("feature_archive/frame_attention.npy", attention_by_frame)


def save_video_confidence(method="attention"):
    if method == "attention":
        conf_scores = np.load("feature_archive/confidence_scores_by_frame.npy")
        conf_score_by_frame = np.mean(conf_scores, axis=2)
        attention_by_frame = np.load("feature_archive/frame_attention.npy")
        confidence = np.mean(conf_score_by_frame * attention_by_frame, axis=1)
        print(confidence.shape)
        np.save("feature_archive/video_confidence_attention.npy", confidence)
    elif method == "average":
        all_uninterp_path = []
        with open(r"feature_archive/all_uninterp_path.txt", "r") as fp:
            for line in fp:
                x = line[:-1]
                all_uninterp_path.append(x)
        assert len(all_uninterp_path) == 951, "Wrong number of videos"
        video_confidence_average = np.zeros(len(all_uninterp_path))
        for i, path in enumerate(all_uninterp_path):
            traj = np.load(path, allow_pickle=True)
            scores = extract_scores(traj["keypoints"])
            video_confidence_average[i] = np.mean(scores)
        print(video_confidence_average.shape)
        np.save("feature_archive/video_confidence_average.npy", video_confidence_average)


def weighted_Logistic(seed=20220712, method="average"):
    file = np.load("feature_archive/all_feature_interp951.npz", allow_pickle=True)
    X_all = file["X"]
    Y_all = file["Y"]

    # split train test
    np.random.seed(seed)
    n_samples = X_all.shape[0]
    test_size = int(n_samples * 0.2)
    perm = np.random.permutation(n_samples)
    test_idx = perm[: test_size]
    train_idx = perm[test_size:]
    X_train = X_all[train_idx]
    Y_train = Y_all[train_idx]
    X_test = X_all[test_idx]
    Y_test = Y_all[test_idx]

    # scale data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # fit unweighted LogisticRegression
    clf_unweighted = LogisticRegression(max_iter=1000, solver="newton-cg")
    clf_unweighted.fit(X_train_scaled, Y_train)
    Y_test_pred_unweighted = clf_unweighted.predict(X_test_scaled)
    acc_unweighted = np.mean(Y_test_pred_unweighted == Y_test)

    # fit weighted LogisticRegression
    confidence = np.load(f"feature_archive/video_confidence_{method}.npy")
    confidence_train = confidence[train_idx]
    clf_weighted = LogisticRegression(max_iter=1000, solver="newton-cg")
    clf_weighted.fit(X_train_scaled, Y_train, sample_weight=confidence_train)
    Y_test_pred_weighted = clf_weighted.predict(X_test_scaled)
    acc_weighted = np.mean(Y_test_pred_weighted == Y_test)

    # print("Unweighted LogisticRegression: {:.2f}%".format(acc_unweighted * 100))
    # print("Weighted LogisticRegression: {:.2f}%".format(acc_weighted * 100))
    return acc_unweighted, acc_weighted


class mydataset_w_index(Dataset):
    def __init__(self, X, Y):
        self.Data = torch.Tensor(X)
        self.Label = torch.LongTensor(Y)
        self.index = torch.LongTensor(np.arange(len(Y)))

    def __getitem__(self, item):
        return self.Data[item], self.Label[item], self.index[item]

    def __len__(self):
        return len(self.Label)


def prepare_data_w_index(test_ratio=0.2, seed=20220712):
    all_data = np.load("feature_archive/all_feature_interp951.npz", allow_pickle=True)
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
    train_dataset = mydataset_w_index(X_train, Y_train)
    test_dataset = mydataset_w_index(X_test, Y_test)
    trainloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    testloader = DataLoader(test_dataset, batch_size=20)

    return trainloader, testloader, train_idx, test_idx


def train_w_iterative_weight(model,
                             epochs,
                             trainloader, train_idx,
                             testloader,
                             optimizer,
                             criterion,
                             save_path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Trained on {}".format(device))
    model.to(device)

    # train model
    weights = np.load("feature_archive/video_confidence_attention.npy")
    weight_train = weights[train_idx]
    best_test_acc = 0
    for epoch in range(epochs):
        for batch in tqdm(trainloader):
            model.train()
            X_batch = batch[0].to(device)
            Y_batch = batch[1].to(device)
            if len(batch) == 3:
                index_batch = batch[2]

            # forward
            logits = model(X_batch)
            weight_batch = torch.Tensor(weight_train[index_batch]).to(device)
            loss = criterion(logits, Y_batch, weight_batch)

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # evaluate
        with torch.no_grad():
            model.eval()
            # evaluate train
            train_loss, train_acc = get_loss_acc(model, trainloader, nn.CrossEntropyLoss())
            # evaluate test
            test_loss, test_acc = get_loss_acc(model, testloader, nn.CrossEntropyLoss())

        print("Epoch{}/{}  train loss: {}  test loss: {}  train acc: {}  testacc: {}".format(epoch + 1, epochs,
                                                                                             train_loss, test_loss,
                                                                                             train_acc, test_acc))

        # save model weights if  it's the best
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), save_path)
            print("Saved")


def weighted_CrossEntropyLoss(logits, y, weights):
    loss_function = nn.CrossEntropyLoss(reduction="none")
    loss_tmp = loss_function(logits, y)
    loss = loss_tmp * weights
    return loss.sum() / weights.sum()


def Transformer_weighted_train():
    seed = 20220728
    experiment_path = "Transformer/weighted_training"
    if not os.path.exists(experiment_path):
        os.mkdir(experiment_path)

    for i in range(5):
        this_seed = seed + i
        trainloader, testloader, train_idx, test_idx = prepare_data_w_index(seed=this_seed)
        model = transformer_huge()
        epochs = 200
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
        criterion = weighted_CrossEntropyLoss
        save_path = experiment_path + f"/Transformer_huge_weighted_{i+1}.pth"
        train_w_iterative_weight(model, epochs,
                                 trainloader, train_idx, testloader,
                                 optimizer, criterion,
                                 save_path)


def evaluate_weighted_Transformer():
    model = transformer_huge().to(device)
    seed = 20220728

    accs_unweighted = []
    accs_weighted = []
    for i in range(5):
        this_seed = seed + i
        trainloader, testloader, train_idx, test_idx = prepare_data_w_index(seed=this_seed)

        model.load_state_dict(torch.load(f"Transformer/experiment_shuffle/no_shuffle/Transformer_huge_no_shuffle_{i+1}.pth",
                                         map_location=device))
        _, acc = get_loss_acc(model, testloader, nn.CrossEntropyLoss())
        accs_unweighted.append(acc)

        model.load_state_dict(
            torch.load(f"Transformer/weighted_training/Transformer_huge_weighted_{i+1}.pth",
                       map_location=device))
        _, acc = get_loss_acc(model, testloader, nn.CrossEntropyLoss())
        accs_weighted.append(acc)

    accs_unweighted = np.array(accs_unweighted)
    accs_weighted = np.array(accs_weighted)
    improvement = accs_weighted - accs_unweighted
    for accs in [accs_unweighted, accs_weighted, improvement]:
        m = np.mean(accs)
        v = np.std(accs)
        print(m, v)


if __name__ == "__main__":
    evaluate_weighted_Transformer()



