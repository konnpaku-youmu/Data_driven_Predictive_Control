import numpy as np
from typing import Tuple, Callable
from scipy import linalg
import matplotlib.pyplot as plt

def forward_euler(A, B, Ts) -> Tuple[np.ndarray]:
    n_states = A.shape[1]

    Ad = np.eye(n_states) + Ts * A
    Bd = Ts * B

    return Ad, Bd

def hankelize(vec: np.ndarray, L: int) -> np.ndarray:
    T = vec.shape[0]
    n = vec.shape[1]
    assert(T >= L)

    H = np.zeros([L*n, T-L+1])

    for i in range(T-L+1):
        H[:, i] = vec[i:i+L, :, :].reshape([L*n])
    
    return H

if __name__ == "__main__":
    v = np.random.rand(115, 2, 1)
    H = hankelize(v, 54)
    print(H.shape)
    plt.matshow(H)
    plt.show()
