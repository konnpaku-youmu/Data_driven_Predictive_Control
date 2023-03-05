import numpy as np
from typing import Tuple, Callable
from scipy import linalg
import matplotlib.pyplot as plt

def forward_euler(A, B, Ts) -> Tuple[np.ndarray]:
    n_states = A.shape[1]

    Ad = np.eye(n_states) + Ts * A
    Bd = Ts * B

    return Ad, Bd

def zoh(A, B, Ts) -> Tuple[np.ndarray]:
    em_upper = np.hstack((A, B))

    # Need to stack zeros under the a and b matrices
    em_lower = np.hstack((np.zeros((B.shape[1], A.shape[0])),
                          np.zeros((B.shape[1], B.shape[1]))))

    em = np.vstack((em_upper, em_lower))
    ms = linalg.expm(Ts * em)

    # Dispose of the lower rows
    ms = ms[:A.shape[0], :]

    Ad = ms[:, 0:A.shape[1]]
    Bd = ms[:, A.shape[1]:]

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
