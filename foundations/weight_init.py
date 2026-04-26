import torch
import math
from typing import List


class Solution:

    def xavier_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        torch.manual_seed(0)
        std = (2 / (fan_in + fan_out)) ** 0.5
        W = torch.randn(fan_out, fan_in) * std
        return [[round(v, 4) for v in row] for row in W.tolist()]

    def kaiming_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        torch.manual_seed(0)
        std = (2 / fan_in) ** 0.5
        W = torch.randn(fan_out, fan_in) * std
        return [[round(v, 4) for v in row] for row in W.tolist()]

    def check_activations(self, num_layers: int, input_dim: int, hidden_dim: int, init_type: str) -> List[float]:
        torch.manual_seed(0)
        weights = []
        for i in range(num_layers):
            fan_in = input_dim if i == 0 else hidden_dim
            fan_out = hidden_dim
            if init_type == 'xavier':
                std = (2 / (fan_in + fan_out)) ** 0.5
            elif init_type == 'kaiming':
                std = (2 / fan_in) ** 0.5
            else:
                std = 1.0
            weights.append(torch.randn(fan_out, fan_in) * std)
        x = torch.randn(input_dim)
        stds = []
        for W in weights:
            x = torch.relu(W @ x)
            stds.append(round(x.std().item(), 2))
        return stds



