import scipy
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import normalize
from .utils import slice_y


class SAVE(BaseEstimator, TransformerMixin):
    def __init__(self, n_directions='auto', n_slices=10, copy=True):
        self.n_directions = n_directions
        self.n_slices = n_slices
        self.copy = copy
        self.summary = {}

    def fit(self, X, y):
        """
        :param X: A nxp matrix. n is the number of samples and p is the number of features
        :param y: A length n vector
        """
        n_directions = self.n_directions

        (n_samples, n_features) = X.shape
        mvec = np.mean(X, axis=0)
        covm = np.cov(X, rowvar=False)
        covm_sqrt_inv = scipy.linalg.inv(scipy.linalg.sqrtm(covm))
        Z = np.dot(X - mvec, covm_sqrt_inv)

        # sort rows of Z with respect to the target y
        Z = Z[np.argsort(y), :]

        slices, counts = slice_y(y, self.n_slices)
        self.n_slices_ = counts.shape[0]

        # construct slice covariance matrices
        M = np.zeros((n_features, n_features))
        self.summary["Vs"] = []
        self.summary["Ms"] = []
        self.summary["M2s"] = []
        for slice_idx in range(self.n_slices_):
            n_slice = counts[slice_idx]

            # center the entries in this slice
            Z_slice = Z[slices == slice_idx, :]
            Z_slice -= np.mean(Z_slice, axis=0)

            # slice covariance matrix
            V_slice = np.dot(Z_slice.T, Z_slice) / n_slice
            M_slice = np.eye(n_features) - V_slice
            M += (n_slice / n_samples) * np.dot(M_slice, M_slice)
            self.summary["Vs"].append(V_slice)
            self.summary["Ms"].append(M_slice)
            self.summary["M2s"].append(np.dot(M_slice, M_slice))
        self.summary["M"] = M

        # eigen-decomposition of slice matrix
        evals, evecs = scipy.linalg.eigh(M)
        evecs = evecs[:, ::-1]  # each column is a eigen-vector
        evals = evals[::-1]

        directions = np.dot(covm_sqrt_inv, evecs)  # each column is a direction
        if self.n_directions == "auto":
            n_directions = np.argmax(np.abs(np.diff(evals))) + 1
        self.n_directions_ = n_directions

        # normalize directions
        directions = normalize(
            directions[:, :self.n_directions_], norm="l2", axis=0
        )
        self.directions_ = directions.T

        self.eigenvalues_ = evals[:self.n_directions_]

        return self

    def transform(self, X):
        return np.dot(X, self.directions_.T)

