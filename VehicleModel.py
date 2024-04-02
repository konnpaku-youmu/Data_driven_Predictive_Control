import numpy as np
from dataclasses import dataclass
import matplotlib.pyplot as plt
import casadi as cs

from SysBase import NonlinearSystem

@dataclass
class VehicleParams():
    m = 190
    I_zz = 110

    lf = 1.22
    lr = 1.22

    Cm = 5000
    Cr0 = 180
    Cr2 = 0.7

    B = 12.56
    C = -1.38
    D = 1.60
    E = -0.58

class RacingCar(NonlinearSystem):
    def __init__(self, x0: np.ndarray, **kwargs) -> None:

        x = cs.SX.sym("x", 8)
        u = cs.SX.sym("u", 2)
        y = cs.SX.sym("y", 6)
        w = cs.SX.sym("w", 8)

        C = np.eye(6, 8)
        
        super().__init__(x, u, y, x0, C, w=w, **kwargs)

        self.params = VehicleParams()

        self.input_names = [r"$\Delta T$", r"$\Delta \delta$"]
        self.output_names = [r"$x$", r"$y$", r"$\psi$", r"$v_x$", r"$v_y$", r"$\omega$"]

        self.state_constraint.lb[-2] = 0
        self.state_constraint.ub[-2] = 1  # maximum drive
        self.state_constraint.lb[-1] = -0.384
        self.state_constraint.ub[-1] = 0.384  # max steering angle (radians)
    
    def _dynamics_num(self, x, u, w) -> cs.SX:
        '''
        x: [x, y, ψ, v_x, v_y, ω, T, δ]
        '''
        px, py, ψ, vx, vy, ω, T, δ = x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7]
        ΔT, Δδ = u[0], u[1]

        m, lf, lr, I_zz = self.params.m, self.params.lf, self.params.lr, self.params.I_zz
        Cm, Cr0, Cr2 = self.params.Cm, self.params.Cr0, self.params.Cr2
        B, C, D = self.params.B, self.params.C, self.params.D

        αf = -cs.atan2(ω*lf + vy, vx) + δ
        αr = cs.atan2(ω*lr - vy, vx)

        F_rx = Cm*T - Cr0 - Cr2 * (vx**2)
        F_fy = D*cs.sin(C*cs.arctan(B*αf))
        F_ry = D*cs.sin(C*cs.arctan(B*αr))
        
        dpx = vx*cs.cos(ψ) - vy*cs.sin(ψ)
        dpy = vx*cs.sin(ψ) + vy*cs.cos(ψ)
        dψ = ω
        dvx = (F_rx - F_fy*cs.sin(δ) + m*vy*ω) / m
        dvy = (F_ry + F_fy*cs.cos(δ) - m*vx*ω) / m
        dω = (F_fy*lf*cs.cos(δ) - F_ry*lr) / I_zz
        dT = ΔT
        dδ = Δδ

        return cs.vertcat(dpx, dpy, dψ, dvx, dvy, dω, dT, dδ)
    