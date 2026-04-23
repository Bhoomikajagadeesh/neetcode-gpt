import numpy as np
from typing import List


class Solution:
    def rms_norm(self, x: List[float], gamma: List[float], eps: float) -> List[float]:
        # Implement RMS Normalization (similar to LayerNorm but without mean centering or beta)
        # Normalize x, then scale by gamma
        # Return result rounded to 4 decimal places as a list
        n=len(x)
        mean_sq=sum(v*v for v in x)/n
        rms=math.sqrt(mean_sq+eps)
        return [round((x[i]/rms)*gamma[i],4) for i in range(n)]


