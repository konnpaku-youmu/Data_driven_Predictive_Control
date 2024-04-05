import numpy as np
import casadi as cs

from SysBase import *

class ActiveSuspension(LinearSystem):
    def __init__(self, x0: np.ndarray, **kwargs) -> None:

        k1 = 35000
        k2 = 190000
        b1 = 1000
        b2 = 2
        m1 = 375
        m2 = 59

        A = np.array([[0.,     1,           0,             -1],
                      [-k1/m1, -b1/m1,      0,          b1/m1],
                      [0,       0,          0,              1],
                      [k1/m2, b1/m2,   -k2/m2,    -(b1+b2)/m2]])

        B = np.array([[0.],
                      [1/m1],
                      [0.],
                      [-1/m2]])

        B2 = np.array([[0.],
                       [0.],
                       [-1.],
                       [b2/m2]])

        C = np.array([[1.,     0.,           0,             0],
                      [-k1/m1, -b1/m1,      0,          b1/m1]])

        D = np.array([[0.],
                      [1/m1]])

        σ_x = np.diag([0, 0, 0, 0])
        σ_y = np.diag([1e-4, 1e-2])

        super().__init__(A, B, B2, C, D, x0,
                         noisy=False, σ_x=σ_x, σ_y=σ_y,
                         **kwargs)
        
        self.output_constraint.lb[0] = -0.127
        self.output_constraint.ub[0] = 0.127
        self.output_constraint.lb[1] = -8
        self.output_constraint.ub[1] = 8

        self.input_constraint.lb[0] = -1000
        self.input_constraint.ub[0] = 1000

        self.input_names = [r"$F$"]
        self.output_names = [r"$\Delta x_{s}$", r"$\ddot{x}_1$"]
