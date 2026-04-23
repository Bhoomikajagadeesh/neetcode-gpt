import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:
    def train(self, X: NDArray[np.float64], y: NDArray[np.float64], epochs: int, lr: float) -> Tuple[NDArray[np.float64], float]:
        # X: (n_samples, n_features)
        # y: (n_samples,) targets
        # epochs: number of training iterations
        # lr: learning rate
        #
        # Model: y_hat = X @ w + b
        # Loss: MSE = (1/n) * sum((y_hat - y)^2)
        # Initialize w = zeros, b = 0
        # return (np.round(w, 5), round(b, 5))
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float)
        n, m = X.shape
        w = np.zeros(m)
        b = 0.0
        for i in range(epochs):
            y_hat = X @ w + b
            error = y_hat - y
            dw = (2/n) * (X.T @ error)
            db = (2/n) * np.sum(error)
            w -= lr * dw
            b -= lr * db
        return np.round(w, 5), round(float(b), 5)

