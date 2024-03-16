import numpy as np
import casadi as cs
from casadi.casadi import sin, cos, tan
import matplotlib.pyplot as plt

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

        B2 = np.array([[0.  ],
                       [0.  ],
                       [-1. ],
                       [b2/m2]])

        C = np.array([[1.,     0.,           0,             0],
                      [-k1/m1, -b1/m1,      0,          b1/m1]])

        D = np.array([[0.],
                      [1/m1]])

        super().__init__(A, B, C, D, x0, B2, **kwargs)
        
        self.output_constraint.lb[0] = -0.127
        self.output_constraint.ub[0] = 0.127
        self.output_constraint.lb[1] = -8
        self.output_constraint.ub[1] = 8

        self.input_constraint.lb[0] = -1000
        self.input_constraint.ub[0] = 1000

        self.output_names = [r"$\Delta x_{s}$", r"$\ddot{x}_1$"]

        self.noisy = False


class BicycleModel(NonlinearSystem):
    def __init__(self, x0: np.ndarray, **kwargs) -> None:
        # super().__init__(states, inputs, outputs, x0, C, **kwargs)
        ...
    


class IPNonlinear(NonlinearSystem):
    def __init__(self, x0, **kwargs) -> None:
        x = cs.MX.sym("x", 4)
        u = cs.MX.sym("u", 1)
        y = cs.MX.sym("y", 2)

        C = np.array([[1, 0, 0, 0],
                      [0, 1, 0, 0]])

        self.lb_output = np.array([[-1.0],
                                   [-0.3]])
        self.ub_output = np.array([[1.0],
                                   [0.3]])
        self.lb_input = np.array([[-6.0]])
        self.ub_input = np.array([[6.0]])

        super().__init__(x, u, y, x0, C, **kwargs)

    def _define_dynamics(self) -> cs.MX:

        Rm = 2.6
        Km = 0.00767
        Kb = 0.00767
        Kg = 3.7
        M = 0.455
        l = 0.305
        m = 0.210
        r = 0.635e-2
        g = 9.81

        kv = (Km*Kg)/(Rm*r)
        kxd = (Km*Kb*(Kg**2))/(Rm*r)

        x0_dot = self._sym_x[2]
        x1_dot = self._sym_x[3]
        x2_dot = kv * self._sym_u - kxd * \
            self._sym_x[2] + m*l*(self._sym_x[3]**2) * \
            sin(self._sym_x[1]) - m*g*sin(self._sym_x[1])
        x2_dot /= M + m*(sin(self._sym_x[1])**2)
        x3_dot = g*sin(self._sym_x[1]) - x2_dot * cos(self._sym_x[1])
        x3_dot /= l

        return cs.vertcat(x0_dot, x1_dot, x2_dot, x3_dot)


class Quadcopter(NonlinearSystem):
    def __init__(self, x0, **kwargs) -> None:
        x = cs.MX.sym('x', 12)
        u = cs.MX.sym('u', 4)
        y = cs.MX.sym('y', 6)

        C = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]])

        super().__init__(x, u, y, x0, C, **kwargs)

    def _define_dynamics(self) -> cs.MX:

        m = 0.5
        l = 0.25
        k = 3e-6
        b = 1e-7
        g = 9.81
        Kd = 0.25
        Ixx = 5e-3
        Iyy = 5e-3
        Izz = 1e-2
        Cm = 1e4

        f_eq = m*g/(4*k*Cm)

        # Vitesse -- direction x,y,z
        vx, vy, vz = self._sym_x[3], self._sym_x[4], self._sym_x[5]
        # Pose
        φ, θ, ψ = self._sym_x[6], self._sym_x[7], self._sym_x[8]
        # Vitesse angulaire
        ωx, ωy, ωz = self._sym_x[9], self._sym_x[10], self._sym_x[11]
        # Entrée du système: Voltage^2
        u1, u2, u3, u4 = self._sym_u[0] + f_eq, self._sym_u[1] + \
            f_eq, self._sym_u[2] + f_eq, self._sym_u[3] + f_eq

        # Définir la dynamique
        ax = (-Kd/m)*vx + (k*Cm/m)*(sin(ψ)*sin(φ) +
                                    cos(ψ)*cos(φ)*sin(θ)) * (u1+u2+u3+u4)
        ay = (-Kd/m)*vy + (k*Cm/m)*(cos(φ)*sin(ψ)*sin(θ) -
                                    cos(ψ)*sin(φ)) * (u1+u2+u3+u4)
        az = (-Kd/m)*vz - g + (k*Cm/m)*cos(θ)*cos(φ) * (u1+u2+u3+u4)

        Φ = ωx + ωy*(sin(φ)*tan(θ)) + ωz*(cos(φ)*tan(θ))
        Θ = ωy * cos(φ) - ωz * sin(φ)
        Ψ = (sin(φ) / cos(θ)) * ωy + (cos(φ) / cos(θ)) * ωz

        Ωx = (l*k*Cm/Ixx) * (u1 - u3) - ((Iyy-Izz)/Ixx)*ωy*ωz
        Ωy = (l*k*Cm/Iyy) * (u2 - u4) - ((Izz-Ixx)/Iyy)*ωx*ωz
        Ωz = (b*Cm/Izz) * (u1 - u2 + u3 - u4) - ((Ixx-Iyy)/Izz)*ωx*ωy

        return cs.vertcat(vx, vy, vz, ax, ay, az, Φ, Θ, Ψ, Ωx, Ωy, Ωz)
