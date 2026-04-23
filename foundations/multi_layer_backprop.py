import numpy as np
from typing import List


class Solution:
    def forward_and_backward(self,
                              x: List[float],
                              W1: List[List[float]], b1: List[float],
                              W2: List[List[float]], b2: List[float],
                              y_true: List[float]) -> dict:
        # Architecture: x -> Linear(W1, b1) -> ReLU -> Linear(W2, b2) -> predictions
        # Loss: MSE = mean((predictions - y_true)^2)
        #
        # Return dict with keys:
        #   'loss':  float (MSE loss, rounded to 4 decimals)
        #   'dW1':   2D list (gradient w.r.t. W1, rounded to 4 decimals)
        #   'db1':   1D list (gradient w.r.t. b1, rounded to 4 decimals)
        #   'dW2':   2D list (gradient w.r.t. W2, rounded to 4 decimals)
        #   'db2':   1D list (gradient w.r.t. b2, rounded to 4 decimals)
        x=np.array(x,dtype=float)
        W1=np.array(W1,dtype=float)
        b1=np.array(b1,dtype=float)
        W2=np.array(W2,dtype=float)
        b2=np.array(b2,dtype=float)
        y_true=np.array(y_true,dtype=float)

        z1=W1@x+b1
        a1=np.maximum(0,z1)
        z2=W2@a1+b2
        y_hat=z2

        loss=np.mean((y_hat-y_true)**2)

        dz2=2*(y_hat-y_true)/len(y_true)
        dW2=np.outer(dz2,a1)
        db2=dz2

        da1=W2.T@dz2
        dz1=da1*(z1>0)
        dW1=np.outer(dz1,x)
        db1=dz1

        dW1=np.round(dW1,4)
        db1=np.round(db1,4)
        dW2=np.round(dW2,4)
        db2=np.round(db2,4)

        dW1[dW1==0]=0
        db1[db1==0]=0
        dW2[dW2==0]=0
        db2[db2==0]=0

        return {
            'loss': round(float(loss),4),
            'dW1': dW1.tolist(),
            'db1': db1.tolist(),
            'dW2': dW2.tolist(),
            'db2': db2.tolist()
        }

