import numpy as np
from dataclasses import dataclass
import matplotlib.pyplot as plt
import casadi as cs

from SysBase import NonlinearSystem

@dataclass
class VehicleParams():
    m = 190

class RacingCar(NonlinearSystem):
    def __init__(self, x0: np.ndarray, **kwargs) -> None:

        x = cs.SX.sym("x", 8)
        u = cs.SX.sym("u", 2)
        y = cs.SX.sym("y", 6)
        w = cs.SX.sym("w", 8)

        C = np.eye(8, 6)
        
        super().__init__(x, u, y, x0, C, w=w, **kwargs)

        self.output_names = [r"$x$", r"$y$", r"$\psi$", r"$v_x$", r"$v_y$", r"$\omega$"]
    
    def _dynamics_num(self, x, u, w) -> cs.SX:
        '''
        x: [x, y, ψ, v_x, v_y, ω]
        '''
        px, py, ψ, vx, vy, ω = x[0], x[1], x[2], x[3], x[4], x[5]
        ΔT, Δδ = u[0], u[1]
        
        dpx = vx * cs.cos(ψ) - vy * cs.sin(ψ)
        dpy = vx * cs.sin(ψ) + vy * cs.cos(ψ)
        dψ = ω
        dvx = ...
        dvy = ...
        dω = ...
        dT = ΔT
        dδ = Δδ

        return cs.vertcat(dpx, dpy, dψ, dvx, dvy, dω, dT, dδ)
