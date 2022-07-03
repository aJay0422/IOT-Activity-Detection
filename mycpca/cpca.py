import numpy as np
import scipy
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import normalize


class CPCA(BaseEstimator, TransformerMixin):
    def __init__(self, n_directions="auto", var_contrib=0.85):
        self.n_directions = n_directions
        self.var_contrib = var_contrib

    def fit(self, X):
        """
        :param X: a list ndarray or ndarray. The shape of X is (n_samples, n_features, T_k)
                  n_samples is the number of samples in the dataset.
                  n_features is the number of features of a time series at each time step.
                  T_k is the sequence length of the kth time series
        """
        n_directions = self.n_directions
        n_samples = len(X)
        n_features = X[0].shape[0]

        # obtain the average of correlation matrix
        C_bar = np.zeros((n_features, n_features))
        for i in range(n_samples):
            smpl = X[i]
            C_i = np.corrcoef(X[i])   # the correlation matrix of the i-th series
            C_bar += C_i
        C_bar /= n_samples

        # eigen-decomposition of the matrix C_bar
        evals, evecs = scipy.linalg.eigh(C_bar)
        evecs = evecs[:, ::-1]   # each column is an eigen-vector
        evals = evals[::-1]   # reverse to get descending order

        directions = evecs
        if self.n_directions == "auto":
            contrib = evals / np.sum(evals)
            n_directions = int(sum(contrib < self.var_contrib) + 1)
        self.n_directions_ = n_directions

        # normalize directions
        directions = normalize(
            directions[:, self.n_directions_], norm="l2", axis=0
        )
        self.directions_ = directions.T   # each row is a direction
        self.eigenvalues_ = evals[:self.n_directions_]

        return self

    def transform(self, X):
        # ndarray dot is performed on the last 2 dimensions
        if isinstance(X, np.ndarray):
            return np.dot(np.transpose(X, (0, 2, 1)), self.directions_.T)
        if isinstance(X, list):
            rtList = []
            for i in range(len(X)):
                rtList.append(np.dot(X[i].transpose(1, 0), self.directions_.T))
            return rtList
