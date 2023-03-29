import numpy as np
from typing import Any, Tuple, Callable
from scipy import linalg
import matplotlib.pyplot as plt

import casadi as cs
from casadi.tools import *

from System import System, LinearSystem, Quadcopter
from helper import hankelize, pagerize, SetpointGenerator


class Controller:
    def __init__(self, model: LinearSystem) -> None:
        self.model = model

    def __call__(self, x: np.ndarray, k: int) -> np.ndarray:
        ...


class OpenLoop(Controller):
    def __init__(self, model: System) -> None:
        super().__init__(model)
        self.u = None

    def set_input_sequence(self, u: np.ndarray) -> None:
        self.u = u

    def generate_rnd_input_seq(self, len: int, lbu: np.ndarray, ubu: np.ndarray, switch_prob: float = 0.3) -> None:
        assert (lbu.shape == ubu.shape)

        self.u = np.ones([len, lbu.shape[0], ubu.shape[1]])

        for k in range(len):
            if np.random.rand() <= switch_prob:
                self.u[k] = np.random.uniform(
                    lbu, ubu, [lbu.shape[0], ubu.shape[1]])
            else:
                self.u[k] = self.u[k-1]

    def __call__(self, x: np.ndarray, k: int) -> np.ndarray:
        try:
            u = self.u[0]
            self.u = np.roll(self.u, -1)
        except IndexError:
            u = np.zeros(self.u[0].shape)
        return u

class CrappyPID(Controller):
    def __init__(self, model: System) -> None:
        super().__init__(model)

        self.I = 0
        self.err_p = 0
    
    def __call__(self, x: np.ndarray, r: int) -> np.ndarray:
        err = x - r
        self.I += err * self.model.Ts
        de = (err - self.err_p) / self.model.Ts

        u = np.zeros([self.model.n_inputs, 1])

        ex, ix, dx = err[0], self.I[0], de[0]
        ey, iy, dy = err[1], self.I[1], de[1]
        ez, iz, dz = err[2], self.I[2], de[2]

        Kx, Ky, Kz = 0.6, 0.6, 16
        Kix, Kiy, Kiz = 0.32, 0.32, 0.25
        Kdx, Kdy, Kdz = 4.5, 4.5, 3.5

        u -= Kz * ez + Kiz * iz + Kdz * dz

        u[0] += Ky * ey + Kiy * iy + Kdy * dy
        u[2] -= Ky * ey + Kiy * iy + Kdy * dy

        u[1] -= Kx * ex + Kix * ix + Kdx * dx
        u[3] += Kx * ex + Kix * ix + Kdx * dx

        ephi, iphi, dphi = err[6], self.I[6], de[6]
        etheta, ith, dth = err[7], self.I[7], de[7]
        epsi, ipsi, dpsi = err[8], self.I[8], de[8]

        Kphi, Ktheta, Kpsi = 7, 4.8, 3.5
        Kiphi, Kitheta, Kipsi = 0.06, 0.15, 0.1
        Kdphi, Kdtheta, Kdpsi = 8.5, 9, 9
        u[0] -= Kphi * ephi + Kdphi * dphi + Kiphi * iphi
        u[2] += Kphi * ephi + Kdphi * dphi + Kiphi * iphi

        u[1] -= Ktheta * etheta + Kdtheta * dth + Kitheta * ith
        u[3] += Ktheta * etheta + Kdtheta * dth + Kitheta * ith

        u[0] -= Kpsi*epsi + Kdpsi * dpsi + Kipsi * ipsi
        u[2] -= Kpsi*epsi + Kdpsi * dpsi + Kipsi * ipsi 
        u[1] += Kpsi*epsi + Kdpsi * dpsi + Kipsi * ipsi 
        u[3] += Kpsi*epsi + Kdpsi * dpsi + Kipsi * ipsi 

        self.err_p = err

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
            self.Q = C.T@C + np.eye(ns, ns) * 1e-2

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

    def __call__(self, x: np.ndarray, r: int) -> np.ndarray:
        return self.K@(x - r)


class MPC(Controller):
    def __init__(self, model: System, **kwargs) -> None:
        super().__init__(model)


class DeePC(Controller):
    def __init__(self, model: System, **kwargs) -> None:
        super().__init__(model)

        kwargs.setdefault("data_mat", "hankel")

        self.T_ini = kwargs["T_ini"]
        self.N = kwargs["N"]
        self.init_ctrl = kwargs["init_law"]
        self.data_mat = kwargs["data_mat"]

        self.exct_bounds = kwargs["exc_bounds"]
        self.exct_states = kwargs["exc_states"]
        self.exct_shapes = kwargs["exc_shapes"]

        try:
            self.Q = kwargs["Q"]
        except KeyError:
            ny = model.n_outputs
            self.Q = np.eye(ny, ny) * 10
        try:
            self.R = kwargs["R"]
        except KeyError:
            nu = model.n_inputs
            self.R = np.eye(nu, nu) * 1e-2

        self.problem = None
        self.solver = None
        self.lbx = None
        self.ubx = None

        self.opt_p = None  # Trajectory constraint: RHS()
        self.opti_vars = None
        self.traj_constraint = None
        self.ref = None    # Tracking reference

    def build_controller(self, **kwargs) -> None:

        L = self.T_ini + self.N
        nx = self.model.n_states
        nu = self.model.n_inputs
        ny = self.model.n_outputs

        kwargs.setdefault("lbx", -np.inf * np.ones([ny, 1]))
        kwargs.setdefault("ubx", np.inf * np.ones([ny, 1]))
        kwargs.setdefault("lbu", -np.inf * np.ones([nu, 1]))
        kwargs.setdefault("ubu", np.inf * np.ones([nu, 1]))

        if self.data_mat == "hankel":
            self.excitation_len = 4*(nu + 1) * (L + nx) - 1
        elif self.data_mat == "page":
            self.excitation_len = 10*L*((nu+1)*(nx+1)-1)

        # Excite the system
        x0 = np.zeros([self.model.n_states, 1])
        sp_gen = SetpointGenerator(
            self.model.n_states, self.excitation_len, self.model.Ts, self.exct_states, self.exct_shapes, self.exct_bounds, switching_prob=0.15)
        self.model.simulate(
            x0, self.excitation_len, control_law=self.init_ctrl, tracking_target=sp_gen())
        self.ref = sp_gen()[:, :self.model.n_outputs]

        self.set_constraints(lb_states=kwargs["lbx"], ub_states=kwargs["ubx"],
                             lb_inputs=kwargs["lbu"], ub_inputs=kwargs["ubu"])
        cost = self.loss()

        self.problem = {"x": self.opti_vars,
                        "f": cost,
                        "g": self.traj_constraint,
                        "p": self.opt_p}

        opts = {"ipopt.tol": 1e-12, "ipopt.max_iter": 200, "ipopt.print_level": 0,
                "expand": True, "verbose": False, "print_time": True}

        self.solver = nlpsol("solver", "ipopt", self.problem, opts)

    def set_constraints(self, lb_states: np.ndarray = None, ub_states: np.ndarray = None,
                        lb_inputs: np.ndarray = None, ub_inputs: np.ndarray = None) -> None:

        L = self.T_ini + self.N

        if self.data_mat == "hankel":
            M_u = hankelize(self.model.u, L)
            M_y = hankelize(self.model.y, L)
        elif self.data_mat == "page":
            M_u = pagerize(self.model.u, L, L//2-2)
            M_y = pagerize(self.model.y, L, L//2-2)

        # Split Hu and Hy into Hp and Hf (ETH paper Eq.5)
        U_p, U_f = np.split(M_u, [self.model.n_inputs * self.T_ini], axis=0)
        Y_p, Y_f = np.split(M_y, [self.model.n_outputs * self.T_ini], axis=0)
        self.Y_f = Y_f
        self.U_f = U_f

        self.opt_p = struct_symMX([entry('u_ini', shape=(self.model.n_inputs), repeat=self.T_ini),
                                   entry('y_ini', shape=(
                                       self.model.n_outputs), repeat=self.T_ini),
                                   entry('ref', shape=(self.model.n_outputs))])
        self.opti_vars = struct_symMX([entry("u", shape=(self.model.n_inputs), repeat=self.N),
                                       entry("y", shape=(
                                           self.model.n_outputs), repeat=self.N),
                                       entry("g", shape=[U_f.shape[1]])])
        self.opti_vars_num = self.opti_vars(0)
        self.opt_p_num = self.opt_p(0)

        if self.model.noisy or type(self.model) != LinearSystem:
            A = vertcat(U_p, Y_p, U_f)
            b = vertcat(*self.opt_p['u_ini'],
                        *self.opt_p['y_ini'],
                        *self.opti_vars['u'])
        else:
            A = vertcat(U_p, Y_p, U_f, Y_f)
            b = vertcat(*self.opt_p['u_ini'],
                        *self.opt_p['y_ini'],
                        *self.opti_vars['u'],
                        *self.opti_vars['y'])

        g = self.opti_vars['g']
        self.traj_constraint = A@g - b

        # input constraints and output constraints
        optim_var = self.opti_vars
        self.lbx = optim_var(-np.inf)
        self.ubx = optim_var(np.inf)
        self.lbx['u'] = lb_inputs
        self.ubx['u'] = ub_inputs
        self.lbx['y'] = lb_states
        self.ubx['y'] = ub_states

    def loss(self) -> cs.MX:
        loss = 0
        Q, R, = self.Q, self.R
        for k in range(self.N):
            y_k = self.opti_vars["y", k] - self.opt_p['ref']
            u_k = self.opti_vars["u", k]
            loss += (1/2) * sum1(y_k.T @ Q @ y_k) + \
                (1/2) * sum1(u_k.T @ R @ u_k)

        if self.model.noisy or type(self.model) != LinearSystem:
            # regularization terms
            λ_s = 15
            g = self.opti_vars["g"]
            Y_f = vertcat(self.Y_f)
            y = vertcat(*self.opti_vars['y'])
            meas_dev = Y_f@g - y
            loss += λ_s * cs.norm_2(meas_dev)**2

            λ_g = 8
            loss += λ_g * cs.norm_2(g)**2

        return loss

    def update_ref(self, r: np.ndarray) -> None:
        self.ref = np.concatenate(
            [self.ref, np.atleast_3d(r.squeeze())], axis=0)

    def __call__(self, x: np.ndarray, r: np.ndarray) -> np.ndarray:

        y_ini = self.model.y[-self.T_ini:].squeeze()
        u_ini = self.model.u[-self.T_ini:].squeeze()
        self.opt_p_num['u_ini'] = vertsplit(u_ini)
        self.opt_p_num['y_ini'] = vertsplit(y_ini)
        self.opt_p_num['ref'] = vertsplit(r.squeeze())
        self.update_ref(r)

        res = self.solver(p=self.opt_p_num, lbg=0, ubg=0,
                          lbx=self.lbx, ubx=self.ubx)

        # Extract optimal solution
        self.opti_vars_num.master = res['x']
        opti_g = self.opti_vars_num['g']

        u = self.U_f @ opti_g

        return u[:self.model.n_inputs]

    def plot_reference(self, **pltargs) -> None:
        pltargs.setdefault('linewidth', 1)

        plot_range = np.linspace(
            0, self.model.y.shape[0]*self.model.Ts, self.model.y.shape[0], endpoint=False)

        for i in range(self.model.n_outputs):
            plt.step(plot_range[:self.excitation_len],
                     self.ref[:self.excitation_len, i, :], **pltargs)
            plt.step(plot_range[self.excitation_len:], self.ref[self.excitation_len:,
                     i, :], label=r"$ref_{}$".format(i), **pltargs)

        plt.legend()
