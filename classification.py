import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.decomposition import PCA

from sliced import SlicedAverageVarianceEstimation
from mysliced.save import SAVE


FEATURE_ARCHIVE = './feature_archive/'


def main():
    # prepare train and test dataset
    train_data = np.load(FEATURE_ARCHIVE + "featured_dataset.npz", allow_pickle=True)
    X_train = train_data["X"]
    Y_train = train_data["Y"]
    test_data = np.load(FEATURE_ARCHIVE + "evaluate_full_feature_interp.npz", allow_pickle=True)
    X_test = test_data["X"]
    Y_test = np.array([1] * 10 + [4] * 10 + [2] * 9 + [3] * 10 + [0] * 10)
    print("Shape of train data and label:", end=" ")
    print(X_train.shape, Y_train.shape)
    print("Shape of test data and label:", end=" ")
    print(X_test.shape, Y_test.shape)
    print("-" * 100)

    # scale data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # fit a baseline model
    parameters = {"C": [0.01, 0.03, 0.1, 0.3, 1, 3]}
    clf = GridSearchCV(estimator=LogisticRegression(max_iter=1000, solver="newton-cg"), param_grid=parameters, cv=5)
    clf.fit(X_train_scaled, Y_train)
    best_C = clf.best_params_["C"]
    lr_base = LogisticRegression(max_iter=1000, solver="newton-cg", C=best_C)
    lr_base.fit(X_train_scaled, Y_train)
    Y_test_pred = lr_base.predict(X_test_scaled)
    test_acc_base = np.mean(Y_test == Y_test_pred)
    test_f1_base = f1_score(Y_test, Y_test_pred, average="macro")
    test_confusion_base = confusion_matrix(Y_test, Y_test_pred)
    print("Baseline Logistic Regression Performance:")
    print("Test accuracy:", test_acc_base)
    print("Test F1 score:", test_f1_base)
    print("Test confusion matrix:")
    print(test_confusion_base)
    print("-" * 100)

    # PCA
    rank = np.linalg.matrix_rank(X_train_scaled)
    pca = PCA(n_components=rank)
    X_train_scaled_pca = pca.fit_transform(X_train_scaled, Y_train)
    X_test_scaled_pca = pca.transform(X_test_scaled)
    print("{} features are reduced to {} features using PCA".format(X_train_scaled.shape[1], rank))
    print("-" * 100)

    # fit a PCA baseline model
    parameters = {"C": [0.01, 0.03, 0.1, 0.3, 1, 3]}
    clf = GridSearchCV(estimator=LogisticRegression(max_iter=1000, solver="newton-cg"), param_grid=parameters, cv=5)
    clf.fit(X_train_scaled_pca, Y_train)
    best_C = clf.best_params_["C"]
    lr_pca_base = LogisticRegression(max_iter=1000, solver="newton-cg", C=best_C)
    lr_pca_base.fit(X_train_scaled_pca, Y_train)
    Y_test_pred = lr_pca_base.predict(X_test_scaled_pca)
    test_acc_pca_base = np.mean(Y_test == Y_test_pred)
    test_f1_pca_base = f1_score(Y_test, Y_test_pred, average="macro")
    test_confusion_pca_base = confusion_matrix(Y_test, Y_test_pred)
    print("PCA Baseline Logistic Regression Performance:")
    print("Test accuracy:", test_acc_pca_base)
    print("Test F1 score:", test_f1_pca_base)
    print("Test confusion matrix:")
    print(test_confusion_pca_base)
    print("-" * 100)

    # mysliced average variance estimation
    save_directions = [i for i in range(10, X_train_scaled_pca.shape[1], 10)]
    test_accs_save = []
    test_f1s_save = []
    for n_directions in save_directions:
        save = SAVE(n_directions=n_directions, n_slices=5)
        X_train_save = save.fit_transform(X_train_scaled_pca, Y_train)
        X_test_save = save.transform(X_test_scaled_pca)
        parameters = {"C": [0.01, 0.03, 0.1, 0.3, 1, 3]}
        clf = GridSearchCV(estimator=LogisticRegression(max_iter=1000, solver="newton-cg"), param_grid=parameters, cv=5)
        clf.fit(X_train_save, Y_train)
        best_C = clf.best_params_["C"]
        lr_save = LogisticRegression(max_iter=1000, solver="newton-cg", C=best_C)
        lr_save.fit(X_train_save, Y_train)
        Y_train_pred = lr_save.predict(X_train_save)
        Y_test_pred = lr_save.predict(X_test_save)
        train_acc_save = np.mean(Y_train == Y_train_pred)
        test_acc_save = np.mean(Y_test == Y_test_pred)
        train_f1_save = f1_score(Y_train, Y_train_pred, average="macro")
        test_f1_save = f1_score(Y_test, Y_test_pred, average="macro")
        test_accs_save.append(test_acc_save)
        test_f1s_save.append(test_f1_save)
        print(
            "Directions:{}   Train ACC:{:.3f}   Test ACC:{:.3f}   Train F1:{:.3f}   Test F1:{:.3f}".format(n_directions,
                                                                                                           train_acc_save,
                                                                                                           test_acc_save,
                                                                                                           train_f1_save,
                                                                                                           test_f1_save))
    print("-" * 100)
    print("Accurate Search")

    argmax_direction = (np.argmax(test_accs_save) + 1) * 10
    direction_upper = np.min([argmax_direction + 10, rank])
    direction_lower = np.max([argmax_direction - 10, 10])
    for n_directions in range(direction_lower, direction_upper + 1):
        save = SlicedAverageVarianceEstimation(n_directions=n_directions, n_slices=5)
        X_train_save = save.fit_transform(X_train_scaled_pca, Y_train)
        X_test_save = save.transform(X_test_scaled_pca)
        parameters = {"C": [0.01, 0.03, 0.1, 0.3, 1, 3]}
        clf = GridSearchCV(estimator=LogisticRegression(max_iter=1000, solver="newton-cg"), param_grid=parameters, cv=5)
        clf.fit(X_train_save, Y_train)
        best_C = clf.best_params_["C"]
        lr_save = LogisticRegression(max_iter=1000, solver="newton-cg", C=best_C)
        lr_save.fit(X_train_save, Y_train)
        Y_train_pred = lr_save.predict(X_train_save)
        Y_test_pred = lr_save.predict(X_test_save)
        train_acc_save = np.mean(Y_train == Y_train_pred)
        test_acc_save = np.mean(Y_test == Y_test_pred)
        train_f1_save = f1_score(Y_train, Y_train_pred, average="macro")
        test_f1_save = f1_score(Y_test, Y_test_pred, average="macro")
        print("Directions:{}   Train ACC:{:.3f}   Test ACC:{:.3f}   Train F1:{:.3f}   Test F1:{:.3f}".
              format(n_directions, train_acc_save, test_acc_save, train_f1_save, test_f1_save))


def test_save():
    # prepare train and test dataset
    train_data = np.load(FEATURE_ARCHIVE + "featured_dataset.npz", allow_pickle=True)
    X_train = train_data["X"]
    Y_train = train_data["Y"]
    test_data = np.load(FEATURE_ARCHIVE + "evaluate_full_feature_interp.npz", allow_pickle=True)
    X_test = test_data["X"]
    Y_test = np.array([1] * 10 + [4] * 10 + [2] * 9 + [3] * 10 + [0] * 10)

    # scale data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # PCA
    rank = np.linalg.matrix_rank(X_train_scaled)
    pca = PCA(n_components=rank)
    X_train_scaled_pca = pca.fit_transform(X_train_scaled, Y_train)
    X_test_scaled_pca = pca.transform(X_test_scaled)

    # mysliced average variance estimation
    save = SlicedAverageVarianceEstimation(n_directions=800, n_slices=5)
    X_train_save = save.fit_transform(X_train_scaled_pca, Y_train)
    X_test_save = save.transform(X_test_scaled_pca)
    lr_save = LogisticRegression(C=1, max_iter=1000, solver="newton-cg")
    lr_save.fit(X_train_save, Y_train)
    Y_test_pred = lr_save.predict(X_test_save)
    print(np.mean(Y_test == Y_test_pred))


if __name__ == "__main__":
   main()
