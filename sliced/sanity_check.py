import numpy as np
import matplotlib.pyplot as plt
from utils import get_data, plot3d, plot1d, get_save_summary
from save import SAVE


def main(n_relevant=1, seed=42):
    X_train, Y_train, X_test, Y_test = get_data(n_relevant=n_relevant, seed=seed)
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
    get_save_summary(save)


if __name__ == "__main__":
    main()
