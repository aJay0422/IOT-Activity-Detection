import numpy as np
import matplotlib.pyplot as plt
from utils import get_data, plot3d, plot1d, get_save_summary, plot_acc
from save import SAVE
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression


def check_save(n_relevant=1, seed=42):
    X_train, Y_train, X_test, Y_test, true_directions = get_data(n_relevant=n_relevant, seed=seed)
    print(X_train.shape, Y_train.shape)
    print(X_test.shape, Y_test.shape)

    # # Visualization
    # plot3d(X_train, Y_train)
    # np.random.seed(seed)
    # true_weights = np.random.rand(n_relevant)
    # true_weights = true_weights / np.sum(true_weights)
    # plot1d(X_train, Y_train, true_weights)

    # sliced average variance estimation
    save = SAVE(n_slices=5)
    save.fit(X_train, Y_train)
    get_save_summary(save, true_directions)


def get_acc(n_relevant=1, n_iter=20, save=None):
    test_accs = []
    for seed in range(1, n_iter + 1):
        X_train, Y_train, X_test, Y_test, true_directions = get_data(n_relevant=n_relevant, seed=seed)
        if save:
            X_train = save.fit_transform(X_train, Y_train)
            X_test = save.transform(X_test)
        parameters = {"C": [0.01, 0.03, 0.1, 0.3, 1, 3]}
        clf = GridSearchCV(estimator=LogisticRegression(solver="newton-cg", max_iter=1000), param_grid=parameters, cv=5)
        clf.fit(X_train, Y_train)
        lr_base = LogisticRegression(solver="newton-cg", C=clf.best_params_["C"], max_iter=1000)
        lr_base.fit(X_train, Y_train)
        Y_test_pred = lr_base.predict(X_test)
        test_acc = np.mean(Y_test == Y_test_pred)
        test_accs.append(test_acc)
    return np.mean(test_accs)


def check(n_relevant=1, n_features=10):
    baseline_acc = get_acc(n_relevant)
    save_accs = []
    for n_directions in range(1, n_features + 1):
        save = SAVE(n_directions=n_directions, n_slices=5)
        save_acc = get_acc(n_relevant, save=save)
        save_accs.append(save_acc)
    plot_acc(save_accs, baseline_acc)


if __name__ == "__main__":
    check()