import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def prepare_data(test_ratio=0.2):
    file = np.load("feature_archive/3D_features_interp951.npz", allow_pickle=True)
    X_all = file["X"]
    Y_all = file["Y"]

    X_train, X_test, Y_train, Y_test = train_test_split(X_all, Y_all, test_size=test_ratio, random_state=42, stratify=Y_all)

    return X_train, X_test, Y_train, Y_test


def experiment1(scale=True):
    X_train, X_test, Y_train, Y_test = prepare_data()
    X_train = X_train.reshape(len(X_train), -1)
    X_test = X_test.reshape(len(X_test), -1)
    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, Y_train)
    Y_test_pred = clf.predict(X_test)
    print(np.mean(Y_test_pred == Y_test))


if __name__ == "__main__":
    experiment1(scale=True)
    experiment1(scale=False)

