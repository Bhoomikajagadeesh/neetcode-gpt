import numpy as np
from typing import Tuple, List


class Solution:
    def batch_norm(self, x: List[List[float]], gamma: List[float], beta: List[float],
                   running_mean: List[float], running_var: List[float],
                   momentum: float, eps: float, training: bool) -> Tuple[List[List[float]], List[float], List[float]]:
        # During training: normalize using batch statistics, then update running stats
        # During inference: normalize using running stats (no batch stats needed)
        # Apply affine transform: y = gamma * x_hat + beta
        # Return (y, running_mean, running_var), all rounded to 4 decimals as lists
        x = np.array(x, float)
        gamma = np.array(gamma, float)
        beta = np.array(beta, float)
        running_mean = np.array(running_mean, float)
        running_var = np.array(running_var, float)
        if training:
            m = np.mean(x, axis=0)
            v = np.mean((x - m) ** 2, axis=0)
            h = (x - m) / np.sqrt(v + eps)
            running_mean = (1 - momentum) * running_mean + momentum * m
            running_var = (1 - momentum) * running_var + momentum * v
        else:
            h = (x - running_mean) / np.sqrt(running_var + eps)
        y = gamma * h + beta
        return (np.round(y, 4).tolist(),np.round(running_mean, 4).tolist(),np.round(running_var, 4).tolist())


