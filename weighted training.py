"""
Train different classifiers with weighted samples
"""
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler

from Transformer.model import transformer_huge
from Transformer.utils import prepare_data
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


if __name__ == "__main__":
    accs = np.zeros((3, 50))
    for i in range(50):
        acc_unweighted, acc_weighted = weighted_Logistic(20220712 + i, method="attention")
        improve = acc_weighted - acc_unweighted
        accs[:, i] = np.array([acc_unweighted, acc_weighted, improve])
        print(str(i) + "finished")
    print(np.sum(accs[2,:] > 0))
    print(np.sum(accs[2,:] < 0))
    m = np.mean(accs, axis=1)
    v = np.std(accs, axis=1)
    print(m)
    print(v)
