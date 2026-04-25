import torch
import torch.nn as nn
from torchtyping import TensorType

class SingleHeadAttention(nn.Module):

    def __init__(self, embedding_dim: int, attention_dim: int):
        super().__init__()
        torch.manual_seed(0)
        # Create three linear projections (Key, Query, Value) with bias=False
        # Instantiation order matters for reproducible weights: key, query, value
        self.k = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.q = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.v = nn.Linear(embedding_dim, attention_dim, bias=False)

    def forward(self, embedded: TensorType[float]) -> TensorType[float]:
        # 1. Project input through K, Q, V linear layers
        # 2. Compute attention scores: (Q @ K^T) / sqrt(attention_dim)
        # 3. Apply causal mask: use torch.tril(torch.ones(...)) to build lower-triangular matrix,
        #    then masked_fill positions where mask == 0 with float('-inf')
        # 4. Apply softmax(dim=2) to masked scores
        # 5. Return (scores @ V) rounded to 4 decimal places
        b, t, s = embedded.shape
        k = self.k(embedded)
        q = self.q(embedded)
        v = self.v(embedded)
        s = q @ k.transpose(1, 2)
        s = s / math.sqrt(k.size(-1))
        m = torch.tril(torch.ones(t, t, device=embedded.device))
        s = s.masked_fill(m == 0, float("-inf"))
        w = torch.softmax(s, dim=2)
        o = w @ v
        return torch.round(o * 10000) / 10000


