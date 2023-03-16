import numpy as np
from typing import Any, Tuple, Callable
from scipy import linalg
import matplotlib.pyplot as plt

import casadi as cs
from casadi.tools import *

from System import LinearSystem
from helper import hankelize, pagerize, SetpointGenerator


class Controller:
    def __init__(self, model: LinearSystem) -> None:
        self.model = model

    def __call__(self, x: np.ndarray, k: int) -> np.ndarray:
        ...


class OpenLoop(Controller):
    def __init__(self, model: LinearSystem) -> None:
        super().__init__(model)
        self.u = None

    def set_input_sequence(self, u: np.ndarray) -> None:
        self.u = u

    def generate_rnd_input_seq(self, len: int, lbu: np.ndarray, ubu: np.ndarray, switch_prob: float = 0.05) -> None:
        assert (lbu.shape == ubu.shape)

        self.u = np.zeros([len, lbu.shape[0], ubu.shape[1]])

        for k in range(len):
            if np.random.rand() <= switch_prob:
                self.u[k] = np.random.uniform(
                    lbu, ubu, [lbu.shape[0], ubu.shape[1]])
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


class DeePC(Controller):
    def __init__(self, model: LinearSystem, **kwargs) -> None:
        super().__init__(model)

        kwargs.setdefault("data_mat", "hankel")

        self.T_ini = kwargs["T_ini"]
        self.N = kwargs["N"]
        self.init_ctrl = kwargs["init_law"]
        self.exct_bounds = kwargs["excitation_bounds"]
        self.data_mat = kwargs["data_mat"]

        try:
            self.Q = kwargs["Q"]
        except KeyError:
            C = model.C
            no = model.n_output
            self.Q = C@C.T + np.eye(no, no) * 1e-2

        try:
            self.R = kwargs["R"]
        except KeyError:
            nu = model.n_inputs
            self.R = np.eye(nu, nu) * 0.1

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

        if self.data_mat == "hankel":
            self.min_exc_len = 2 * (nu + 1) * (L + nx) - 1
        elif self.data_mat == "page":
            self.min_exc_len = L*((nu*L+1)*(nx+1)-1)

        # Excite the system
        x0 = np.zeros([self.model.n_states, 1])
        sp_gen = SetpointGenerator(
            self.model.n_states, self.min_exc_len, self.model.Ts, 0, "rand", self.exct_bounds, switching_prob=0.1)
        self.model.simulate(
            x0, self.min_exc_len, control_law=self.init_ctrl, tracking_target=sp_gen())
        self.ref = sp_gen()[:, :2]

        self.set_constraints()
        cost = self.loss()

        self.problem = {"x": self.opti_vars,
                        "f": cost,
                        "g": self.traj_constraint,
                        "p": self.opt_p}

        opts = {"ipopt.tol": 1e-12, "ipopt.max_iter": 100, "ipopt.print_level": 0,
                "expand": True, "verbose": False, "print_time": True}

        self.solver = nlpsol("solver", "ipopt", self.problem, opts)

    def set_constraints(self) -> None:

        L = self.T_ini + self.N

        if self.data_mat == "hankel":
            M_u = hankelize(self.model.u, L)
            M_y = hankelize(self.model.y, L)
        elif self.data_mat == "page":
            M_u = pagerize(self.model.u, L)
            M_y = pagerize(self.model.y, L)

        # Split Hu and Hy into Hp and Hf (ETH paper Eq.5)
        U_p, U_f = np.split(M_u, [self.model.n_inputs * self.T_ini], axis=0)
        Y_p, Y_f = np.split(M_y, [self.model.n_output * self.T_ini], axis=0)
        self.Y_f = Y_f
        self.U_f = U_f

        self.opt_p = struct_symMX([entry('u_ini', shape=(self.model.n_inputs), repeat=self.T_ini),
                                   entry('y_ini', shape=(
                                       self.model.n_output), repeat=self.T_ini),
                                   entry('ref', shape=(self.model.n_output))])
        self.opti_vars = struct_symMX([entry("u", shape=(self.model.n_inputs), repeat=self.N),
                                       entry("y", shape=(
                                           self.model.n_output), repeat=self.N),
                                       entry("g", shape=[U_f.shape[1]])])
        self.opti_vars_num = self.opti_vars(0)
        self.opt_p_num = self.opt_p(0)

        if self.model.noisy:
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
        self.lbx['u'] = -5.0
        self.ubx['u'] = 5.0
        self.lbx['y'] = np.array([[-1], [-0.3]])
        self.ubx['y'] = np.array([[1], [0.3]])

    def loss(self) -> cs.MX:
        loss = 0
        Q, R, = self.Q, self.R
        for k in range(self.N):
            y_k = self.opti_vars["y", k] - self.opt_p['ref']
            u_k = self.opti_vars["u", k]
            loss += sum1(y_k.T @ Q @ y_k) + sum1(u_k.T @ R @ u_k)

        if self.model.noisy:
            # regularization terms
            λ_s = 250
            g = self.opti_vars["g"]
            Y_f = vertcat(self.Y_f)
            y = vertcat(*self.opti_vars['y'])
            meas_dev = Y_f@g - y
            loss += λ_s * cs.norm_2(meas_dev)**2

            λ_g = 12
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

        return u[0]

    def plot_reference(self, **pltargs) -> None:
        pltargs.setdefault('linewidth', 1)

        plot_range = np.linspace(
            0, self.model.y.shape[0]*self.model.Ts, self.model.y.shape[0], endpoint=False)

        for i in range(self.model.n_output):
            plt.step(plot_range[:self.min_exc_len],
                     self.ref[:self.min_exc_len, i, :], **pltargs)
            plt.step(plot_range[self.min_exc_len:], self.ref[self.min_exc_len:,
                     i, :], label=r"$ref_{}$".format(i), **pltargs)

        plt.legend()
