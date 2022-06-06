import numpy as np
import scipy
import matplotlib.pyplot as plt


def slice_y(y, n_slices=10):
    unique_y_value, counts = np.unique(y, return_counts=True)
    cumsum_y = np.cumsum(counts)
    slice_partition = np.hstack((0, cumsum_y))

    n_y_values = unique_y_value.shape[0]
    if n_y_values < n_slices:
        raise ValueError("number of slices should be smaller  than the number of unique values in y")
    elif n_y_values > n_slices:
        raise ValueError("Regression problem not supported yet")

    slice_indicator = np.ones(y.shape[0], dtype=int)
    for j, (start_idx, end_idx) in enumerate(zip(slice_partition, slice_partition[1:])):
        if j == len(slice_partition) - 2:
            slice_indicator[start_idx:] = j
        else:
            slice_indicator[start_idx: end_idx] = j

    slice_counts = np.bincount(slice_indicator)
    return slice_indicator, slice_counts


def plot3d(X, y):
    fig = plt.figure(figsize=(15, 10))
    ax = plt.axes(projection='3d')
    for cla in range(5):
        idx = (y == cla)
        xdata = X[idx, 0]
        ydata = X[idx, 1]
        zdata = y[idx]
        ax.scatter3D(xdata, ydata, zdata, s=10)
        ax.set_xlabel("X_1", fontsize=15)
        ax.set_ylabel("X_2", fontsize=15)
        ax.set_zlabel("Label", fontsize=15, rotation=0)
        ax.zaxis.set_rotate_label(False)
    plt.show()


def plot1d(X, y, weights):
    n_relevant = len(weights)
    x = np.sum(X[:, :n_relevant] * weights, axis=1)
    fig = plt.figure(figsize=(15, 3))
    ax = plt.axes()
    for cla in range(5):
        idx = (y == cla)
        xdata = x[idx]
        ydata = y[idx]
        ax.scatter(xdata, ydata, s=1)
    plt.show()


def get_save_summary(save):
    summary = save.summary
    print("Learned dirsctions: ", save.directions_)
    print()

    for cla in range(5):
        M = summary["Ms"][cla]
        M_evals, _ = scipy.linalg.eigh(M)
        M_evals = M_evals[::-1]
        M2 = summary["M2s"][cla]
        M2_evals, _ = scipy.linalg.eigh(M2)
        M2_evals = M2_evals[::-1]
        print("Class Label: {}".format(cla))

        print("Eigenvalues of (I-V):", M_evals)
        print("Ratio of the 1st and 2nd eigenvalues:", M_evals[0] / M_evals[1])
        print("Eigenvalues of (I-V)^2", M2_evals)
        print("Ratio of the 1st and 2nd eigenvalues:", M2_evals[0] / M2_evals[1])

        print("------------------------------")
    M = summary["M"]
    evals, _ = scipy.linalg.eigh(M)
    evals = evals[::-1]
    print("Eigenvalues of M:", evals)
    print("Ratio of the 1st and 2nd eigenvalues:", evals[0] / evals[1])


def get_data(n_features=10, n_relevant=3, n_samples=10000, seed=42, mode='tt'):
    np.random.seed(seed)
    X = np.random.randn(n_samples, n_features)

    def get_label(X, n_relevant, seed=seed):
        np.random.seed(seed)
        weights = np.random.rand(n_relevant)
        weights = weights / sum(weights)
        x_sufficient = np.sum(X[:, :n_relevant] * weights, axis=1)
        x_sufficient = np.sin(x_sufficient * 0.2) * 5 + np.random.random(X.shape[0])
        q = np.quantile(x_sufficient, q=(0.2, 0.4, 0.6, 0.8))
        y = np.zeros(n_samples)

        y[(x_sufficient <= q[0])] = 0
        y[((q[0] < x_sufficient) * (x_sufficient <= q[1]))] = 1
        y[((q[1] < x_sufficient) * (x_sufficient <= q[2]))] = 2
        y[((q[2] < x_sufficient) * (x_sufficient <= q[3]))] = 3
        y[(q[3] < x_sufficient)] = 4
        return y

    Label = get_label(X, n_relevant)
    perm = np.random.permutation(n_samples)

    if mode == 'tt':
        train_idx = perm[:int(n_samples * 0.8)]
        test_idx = perm[int(n_samples * 0.8):]
        return X[train_idx, :], Label[train_idx], X[test_idx, :], Label[test_idx]
    elif mode == 'tvt':
        train_idx = perm[:int(n_samples * 0.8)]
        valid_idx = perm[int(n_samples * 0.8):int(n_samples * 0.9)]
        test_idx = perm[int(n_samples * 0.9):]
        return X[train_idx, :], Label[train_idx], X[valid_idx, :], Label[valid_idx], X[test_idx, :], Label[test_idx]