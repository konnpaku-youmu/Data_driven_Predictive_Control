from typing import Tuple
import numpy as np
from numpy.linalg import *
import casadi as ca


def get_system_dynamics() -> Tuple[np.ndarray, np.ndarray]:
    Rm = 2.6
    Km = 0.00767
    Kb = 0.00767
    Kg = 3.7
    M = 0.455
    l = 0.305
    m = 0.210
    r = 0.635e-2
    g = 9.81

    A = np.array([[0, 0, 1, 0],
                  [0, 0, 0, 1],
                  [0, -m * g / M, -(Kg**2 * Km * Kb) / (M * Rm * r**2), 0],
                  [0, (M + m) * g / (M * l), (Kg**2 * Km * Kb) / (M * Rm * r**2 * l), 0]])

    B = np.array([[0],
                  [0],
                  [(Km * Kg) / (M * Rm * r)],
                  [(-Km * Kg) / (r * Rm * M * l)]])

    return A, B


def discretize_dynamics(A: np.ndarray, B: np.ndarray, Ts: int) -> Tuple[np.ndarray, np.ndarray]:
    n_inputs = np.size(A, 1)
    n_ctrls = np.size(B, 1)

    Ad = np.eye(n_inputs) + Ts * A
    Bd = Ts * B

    return Ad, Bd


def main():
    A, B = get_system_dynamics()

    T_s = 0.05
    Ad, Bd = discretize_dynamics(A, B, T_s)

    x = ca.SX.sym('x')
    v = ca.SX.sym('v')
    a = ca.SX.sym('a')
    w = ca.SX.sym('w')

    states = ca.vertcat(x,
                        v,
                        a,
                        w)
    
    u = ca.SX.sym('u')
    controls = ca.vertcat(u)

    rhs = Ad @ states + Bd @ controls
    f = ca.Function('f', [states, controls], [])
    
    Q = np.diag([0.5, 2, 0.1, 0.1])
    R = 0.05

    

if __name__ == "__main__":
    main()
