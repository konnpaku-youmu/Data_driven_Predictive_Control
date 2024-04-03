import numpy as np
from dataclasses import dataclass
from typing import Callable, Any
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle
import casadi as cs

from SysBase import NonlinearSystem


@dataclass
class VehicleParams():
    m = 0.041
    I_zz = 27.8e-6

    lf = 0.029
    lr = 0.033

    Cm1 = 0.287
    Cm2 = 0.0545
    Cr0 = 0.0518
    Cr2 = 0.00035

    Bf = 2.579
    Cf = 1.2
    Df = 0.192

    Br = 3.3852
    Cr = 1.2691
    Dr = 0.1737

    E = 0.58


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

        self.state_constraint.lb[3] = 0
        self.state_constraint.ub[3] = 10
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
        Cm1, Cm2, Cr0, Cr2 = self.params.Cm1, self.params.Cm2, self.params.Cr0, self.params.Cr2
        Bf, Cf, Df = self.params.Bf, self.params.Cf, self.params.Df
        Br, Cr, Dr = self.params.Br, self.params.Cr, self.params.Dr

        αf = -cs.atan2(ω*lf + vy, vx) + δ
        αr = cs.atan2(ω*lr - vy, vx)

        F_rx = (Cm1 - Cm2*vx)*T - Cr0 - Cr2 * (vx**2)
        F_fy = Df*cs.sin(Cf*cs.arctan(Bf*αf))
        F_ry = Dr*cs.sin(Cr*cs.arctan(Br*αr))

        dpx = vx*cs.cos(ψ) - vy*cs.sin(ψ)
        dpy = vx*cs.sin(ψ) + vy*cs.cos(ψ)
        dψ = ω
        dvx = (F_rx - F_fy*cs.sin(δ) + m*vy*ω) / m
        dvy = (F_ry + F_fy*cs.cos(δ) - m*vx*ω) / m
        dω = (F_fy*lf*cs.cos(δ) - F_ry*lr) / I_zz
        dT = ΔT
        dδ = Δδ

        return cs.vertcat(dpx, dpy, dψ, dvx, dvy, dω, dT, dδ)

    def plot_phasespace(self,
                        axis: plt.Axes,
                        *,
                        states: list,
                        trim_exci: bool = False,
                        **pltargs):
        y = self.get_y()

        super().plot_phasespace(axis=axis, states=states,
                                trim_exci=trim_exci, 
                                colormap=y[:, 3, 0],
                                **pltargs)

        l = 1
        w = 0.5 * l

        for i in range(0, self.n_steps, 20):
            vehicle = Rectangle(y[i, :2, :] - np.array([[l/2], [w/2]]), l, w,
                                angle=180*y[i, 2, :]/np.pi,
                                rotation_point='center')
            axis.add_patch(vehicle)

        return
