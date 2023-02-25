import numpy as np
from typing import Any, Tuple, Callable
from scipy import linalg

from System import LinearSystem


class Controller:
    def __init__(self, model: Any) -> None:
        self.model = model

    def __call__(self, x: np.ndarray, k: int) -> np.ndarray:
        ...


class OpenLoop(Controller):
    def __init__(self, model: Any) -> None:
        super().__init__(model)
        self.u = None

    def set_input_sequence(self, u:np.ndarray) -> None:
        self.u = u
    
    def generate_rnd_input_seq(self, len:int, lbu:np.ndarray, ubu:np.ndarray, switch_prob: float = 0.05) -> None:
        assert(lbu.shape == ubu.shape)

        self.u = np.zeros([len, lbu.shape[0], ubu.shape[1]])

        for k in range(len):
            if np.random.rand() <= switch_prob:
                self.u[k] = np.random.uniform(lbu, ubu, [lbu.shape[0], ubu.shape[1]])
            else:
                self.u[k] = self.u[k-1]
    
    def __call__(self, x: np.ndarray, k: int) -> np.ndarray:
        try:
            u = self.u[k]
        except IndexError:
            u = np.zeros(self.u[0].shape)
        return u


class LQRController(Controller):
    def __init__(self, model: LinearSystem, **kwargs) -> None:
        super().__init__(model)

        self.K = None

        try:
            self.Q = kwargs["Q"]
        except KeyError:
            C = model.C
            ns = model.n_states
            self.Q = C.T@C + np.eye(ns, ns) * 1e-3

        try:
            self.R = kwargs["R"]
        except KeyError:
            nu = model.n_inputs
            self.R = np.eye(nu, nu) * 0.1

        self.compute_K()

    def compute_K(self) -> None:
        A = self.model.A
        B = self.model.B

        P = linalg.solve_discrete_are(A, B, self.Q, self.R)
        self.K = -np.linalg.inv(self.R + B.T@P@B)@B.T@P@A

    def __call__(self, x: np.ndarray, k: int) -> np.ndarray:
        return self.K@x


class SetpointGenerator:
    def __init__(self, n_step) -> None:
        ...
    
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        pass
