import numpy as np
from SysBase import LinearSystem


class Estimator:
    def __init__(self) -> None:
        pass

    def __call__(self, y: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class KF(Estimator):
    def __init__(self, model: LinearSystem, x0_est, σ_w, σ_v, σ_p) -> None:
        super().__init__()

        self.A = model.A
        self.B = model.B
        self.C = model.C
        self.f, self.h = model._f, model._output

        self.x_hat = x0_est
        self.x_bar = x0_est

        self.Q, self.R = np.eye(model.n) * σ_w**2, np.eye(model.p) * σ_v**2
        self.Pk = np.eye(model.n) * σ_p**2

        C, Pk, R = self.C, self.Pk, self.R
        self.Lk = Pk @ C.T @ np.linalg.inv(C@Pk@C.T + R)

    def __call__(self, y: np.ndarray) -> np.ndarray:

        A, C = self.A, self.C

        Q, Pk_m = self.Q, self.Pk

        self.Lk = Pk_m @ C.T @ np.linalg.inv(C@Pk_m@C.T + self.R)
        self.Pk = Pk_m - self.Lk @ C @ Pk_m
        self.x_hat = self.x_bar + self.Lk @ (y - C@self.x_bar)

        self.x_bar = A @ self.x_hat
        self.Pk = A @ self.Pk @ A.T + self.Q

        return self.x_hat


class EKF(Estimator):
    def __init__(self) -> None:
        super().__init__()
    
    def __call__(self, y: np.ndarray) -> np.ndarray:
        return super().__call__(y)


class MHE(Estimator):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, y: np.ndarray) -> np.ndarray:
        return super().__call__(y)
