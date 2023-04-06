import numpy as np
from typing import Tuple, Callable
from scipy import linalg
import casadi as cs
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

def runge_kutta(order: int, f: cs.Function):
    
    ...

def hankelize(vec: np.ndarray, L: int) -> np.ndarray:
    T = vec.shape[0]
    n = vec.shape[1]
    assert(T >= L)

    H = np.zeros([L*n, T-L+1])

    for i in range(T-L+1):
        H[:, i] = vec[i:i+L, :, :].reshape([L*n])
    
    return H

def pagerize(vec: np.ndarray, L: int, S: int = None) -> np.ndarray:
    N=vec.shape[0]
    n=vec.shape[1]
    if S > L:
        print("Stride larger than L. Reset to L")
        S = L

    k = (N-L)//S + 1

    if (N-L) % S != 0:
        N = (k-1) * S + L
        vec = vec[:N]
    
    P = np.zeros([L*n, k])

    for i in range(k):
        P[:, i] = vec[i*S:i*S+L, :, :].reshape([L*n])

    return P

class SetpointGenerator:
    def __init__(self, n_output, n_steps, Ts,
                 trac_states: list, shapes: list,
                 bounds: np.ndarray, **kwargs) -> None:

        self.Ts = Ts
        self.n_output = n_output
        self.sim_steps = n_steps
        self.sp = np.zeros([n_steps, n_output, 1])

        if isinstance(trac_states, int) and isinstance(shapes, str):
            trac_states = [trac_states]
            shapes = [shapes]

        assert len(trac_states) == len(shapes), "{0}, {1}".format(len(trac_states), len(shapes))
        assert bounds.shape[0] == len(trac_states), "{0}, {1}".format(bounds.shape, len(trac_states))

        for state, shape, bound in zip(trac_states, shapes, bounds):
            sp_state = np.zeros([n_steps, 1])

            if shape == "ramp":
                pass
            elif shape == "step":
                kwargs.setdefault("step_time", int(1/self.Ts))
                kwargs.setdefault("height", 1)
                step_time = kwargs["step_time"]
                height = kwargs["height"]
                sp_state[step_time:] = height
            elif shape == "rand":
                kwargs.setdefault("switch_prob", 0.05)
                switching_prob = kwargs["switch_prob"]

                for k in range(n_steps):
                    if np.random.rand() <= switching_prob:
                        sp_state[k] = np.random.uniform(
                            np.min(bound), np.max(bound))
                    else:
                        sp_state[k] = sp_state[k-1]

            self.sp[:, state, :] = sp_state

    def plot(self, **kwargs) -> None:
        sim_range = np.linspace(
            0, self.sim_steps*self.Ts, self.sim_steps, endpoint=False)
        for i in range(self.n_output):
            plt.step(sim_range, self.sp[:, i, :], **kwargs)

    def __call__(self) -> np.ndarray:
        return self.sp

if __name__ == "__main__":
    v = np.linspace([1, 1], [10, 10], 50)
    H = hankelize(np.atleast_3d(v), 5)
    P = pagerize(np.atleast_3d(v), 5, 2)
    print(P.shape, H.shape)
    plt.matshow(P)
    plt.show()
    