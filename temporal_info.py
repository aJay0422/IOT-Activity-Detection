import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


def fit_and_eval(X_train, Y_train, X_test, Y_test):
    X_train = X_train.reshape(len(X_train), -1)
    X_test = X_test.reshape(len(X_test), -1)
    clf = LogisticRegression(max_iter=10000, random_state=42, solver="saga")
    clf.fit(X_train, Y_train)
    Y_test_pred = clf.predict(X_test)
    acc = np.mean(Y_test_pred == Y_test)
    print("Accuracy: {:.4f}%".format(acc * 100))


file = np.load("feature_archive/all_feature_interp951.npz", allow_pickle=True)
X_all = file["X"]
Y_all = file["Y"]

# reshape data to (n_samples, n_features, n_frames)
X_all = X_all.reshape(-1, 100, 34).transpose(0, 2, 1)

X_train, X_test, Y_train, Y_test = train_test_split(X_all, Y_all, test_size=0.2, random_state=42, stratify=Y_all)

# without shuffle
print("Without shuffle:")
fit_and_eval(X_train, Y_train, X_test, Y_test)

# shuffle with different seeds
print("Shuffle the order of frames with 5 different seeds: ")
for seed in [42, 43, 44, 45, 46]:
    np.random.seed(seed)
    perm = np.random.permutation(100)
    X_train = X_train[:, :, perm]
    X_test = X_test[:, :, perm]
    fit_and_eval(X_train, Y_train, X_test, Y_test)

