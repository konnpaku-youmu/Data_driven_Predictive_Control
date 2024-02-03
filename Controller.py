import numpy as np
from enum import Enum
from scipy import linalg
import matplotlib.pyplot as plt

import casadi as cs
from casadi.tools import *

from SysModels import System, LinearSystem
from helper import hankelize, pagerize, RndSetpoint


class SMStruct(Enum):
    HANKEL = 0
    PARTIAL_HANKEL = 1
    PAGE = 2


class OCPType(Enum):
    CANONICAL = 0
    REGULARIZED = 1


class Controller:
    def __init__(self, model: System, **kwargs) -> None:
        self.model = model
        self.closed_loop = None
        self.u = None

    def __call__(self, x: np.ndarray, r: np.ndarray) -> np.ndarray:
        raise NotImplementedError()


class OpenLoop(Controller):
    def __init__(self, model: System) -> None:
        super().__init__(model)
        self.closed_loop = False

    @classmethod
    def given_input_seq(cls, model: System, u: np.ndarray):
        inst = cls.__new__(cls)
        super(OpenLoop, inst).__init__(model=model)
        inst.__set_input_sequence(u=u)
        return inst

    @classmethod
    def rnd_input(cls, model: System, length: int):
        inst = cls.__new__(cls)
        super(OpenLoop, inst).__init__(model=model)
        inst.__rnd_input_seq(
            length=length, lbu=model.lb_input, ubu=model.ub_input)
        return inst

    def __set_input_sequence(self, u: np.ndarray) -> None:
        self.u = u

    def __rnd_input_seq(self, length: int, lbu: np.ndarray, ubu: np.ndarray, switch_prob: float = 0.5) -> None:
        assert (lbu.shape == ubu.shape)

        self.u = np.ones([length, lbu.shape[0], ubu.shape[1]])

        for k in range(length):
            if np.random.rand() <= switch_prob:
                self.u[k] = np.random.uniform(
                    lbu, ubu, [lbu.shape[0], ubu.shape[1]])
            else:
                self.u[k] = self.u[k-1]

    def __call__(self, x: np.ndarray, k: int) -> np.ndarray:
        u = self.u[0]
        self.u = np.roll(self.u, -1)
        return u, None


class CrappyPID(Controller):
    def __init__(self, model: System) -> None:
        super().__init__(model)

        self.closed_loop = True
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

        return u, None


class LQRController(Controller):
    def __init__(self, model: LinearSystem, **kwargs) -> None:
        super().__init__(model)

        self.closed_loop = True
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
        _r = np.zeros((2, 1))
        r = np.vstack([r, _r])
        return self.K@(x - r), None


class MPC(Controller):
    def __init__(self, model: System, **kwargs) -> None:
        super().__init__(model)


class DeePC(Controller):
    def __init__(self, model: System, T_ini: int, horizon: int,
                 init_law: Controller, data_mat: SMStruct = SMStruct.HANKEL, **kwargs) -> None:
        super().__init__(model)

        kwargs.setdefault("lbx", self.model.lb_output)
        kwargs.setdefault("ubx", self.model.ub_output)
        kwargs.setdefault("lbu", self.model.lb_input)
        kwargs.setdefault("ubu", self.model.ub_input)
        kwargs.setdefault("λ_s", 0)
        kwargs.setdefault("λ_g", 0)

        kwargs.setdefault("Q", np.eye(
            self.model.n_outputs, self.model.n_outputs) * 10)
        kwargs.setdefault("R", np.eye(
            self.model.n_inputs, self.model.n_inputs) * 1e-2)

        self.closed_loop = True

        self.T_ini = T_ini
        self.horizon = horizon
        self.init_ctrl = init_law
        self.sm_struct = data_mat

        self.Q = kwargs["Q"]
        self.R = kwargs["R"]
        self.λ_s = kwargs["λ_s"]
        self.λ_g = kwargs["λ_g"]

        self.problem = None
        self.solver = None
        self.lbx = None
        self.ubx = None

        self.opt_p = None  # Trajectory constraint: RHS()
        self.opti_vars = None
        self.traj_constraint = None
        self.ref = None    # Tracking reference

        self.objective = []

        self.__build_controller(**kwargs)

    def __build_controller(self, **kwargs) -> None:
        L = self.T_ini + self.horizon
        nx = self.model.n_states
        nu = self.model.n_inputs
        ny = self.model.n_outputs

        if self.sm_struct == SMStruct.HANKEL:
            self.init_len = (nu + 1) * (L + nx) - 1
        elif self.sm_struct == SMStruct.PAGE:
            self.init_len = 2*L*((nu+1)*(nx+1)-1)

        self.ref = np.zeros([self.init_len, ny, 1])
        if self.init_ctrl.closed_loop:
            cloop_sp = RndSetpoint(ny, self.init_len, 0,
                                   np.array([[[-1], [1]]]))
            self.ref = cloop_sp()

        self.model.simulate(
            self.init_len, control_law=self.init_ctrl, ref_traj=self.ref)

        self.__set_constraints(lb_states=kwargs["lbx"], ub_states=kwargs["ubx"],
                               lb_inputs=kwargs["lbu"], ub_inputs=kwargs["ubu"])
        cost = self.loss()

        self.problem = {"x": self.opti_vars,
                        "f": cost,
                        "g": self.traj_constraint,
                        "p": self.opt_p}

        opts = {"ipopt.tol": 1e-12,
                "ipopt.max_iter": 200,
                "ipopt.print_level": 0,
                "expand": True,
                "verbose": False,
                "print_time": True}

        self.solver = nlpsol("solver", "ipopt", self.problem, opts)

    def __set_constraints(self, lb_states: np.ndarray = None, ub_states: np.ndarray = None,
                          lb_inputs: np.ndarray = None, ub_inputs: np.ndarray = None) -> None:

        L = self.T_ini + self.horizon

        if self.sm_struct == SMStruct.HANKEL:
            M_u = hankelize(self.model.get_u(), L)
            M_y = hankelize(self.model.get_y(), L)
        elif self.sm_struct == SMStruct.PAGE:
            M_u = pagerize(self.model.get_u(), L, L//2-2)
            M_y = pagerize(self.model.get_y(), L, L//2-2)

        # Split Hu and Hy into Hp and Hf (ETH paper Eq.5)
        U_p, U_f = np.split(M_u, [self.model.n_inputs * self.T_ini], axis=0)
        Y_p, Y_f = np.split(M_y, [self.model.n_outputs * self.T_ini], axis=0)
        self.Y_f = Y_f
        self.U_f = U_f

        self.opt_p = struct_symMX([entry('u_ini', shape=(self.model.n_inputs), repeat=self.T_ini),
                                   entry('y_ini', shape=(
                                       self.model.n_outputs), repeat=self.T_ini),
                                   entry('ref', shape=(self.model.n_outputs))])
        self.opti_vars = struct_symMX([entry("u", shape=(self.model.n_inputs), repeat=self.horizon),
                                       entry("y", shape=(
                                           self.model.n_outputs), repeat=self.horizon),
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

        for k in range(self.horizon):
            y_k = self.opti_vars["y", k] - self.opt_p['ref']
            u_k = self.opti_vars["u", k]
            loss += (1/2) * sum1(y_k.T @ Q @ y_k) + \
                    (1/2) * sum1(u_k.T @ R @ u_k)

        if self.model.noisy or type(self.model) != LinearSystem:
            # regularization terms
            g = self.opti_vars["g"]
            Y_f = vertcat(self.Y_f)
            y = vertcat(*self.opti_vars['y'])
            u_k = self.opti_vars["u", k]
            esti_err = Y_f@g - y
            loss += self.λ_s * (cs.norm_2(esti_err)**2 + sum1(u_k.T @ R @ u_k))
            loss += self.λ_g * cs.norm_2(g)**2

        return loss

    def __update_ref(self, r: np.ndarray) -> None:
        self.ref = np.concatenate(
            [self.ref, np.atleast_3d(r.squeeze())], axis=0)

    def __call__(self, x: np.ndarray, r: np.ndarray) -> np.ndarray:
        y_ini = self.model.get_y()[-self.T_ini:].squeeze()
        u_ini = self.model.get_u()[-self.T_ini:].squeeze()

        self.opt_p_num['u_ini'] = vertsplit(u_ini)
        self.opt_p_num['y_ini'] = vertsplit(y_ini)
        self.opt_p_num['ref'] = vertsplit(r.squeeze())
        self.__update_ref(r)

        res = self.solver(p=self.opt_p_num, lbg=0, ubg=0,
                          lbx=self.lbx, ubx=self.ubx)

        # Extract optimal solution
        self.opti_vars_num.master = res['x']
        loss_val = res['f']
        opti_g = self.opti_vars_num['g']

        self.objective.append(loss_val)

        u = self.U_f @ opti_g


        return u[:self.model.n_inputs], u[self.model.n_inputs:]

    def plot_reference(self, axis = None, **pltargs) -> None:
        pltargs.setdefault('linewidth', 1)

        y = self.model.get_y()
        plot_range = np.linspace(
            0, y.shape[0]*self.model.Ts, y.shape[0], endpoint=False)

        for i in range(self.model.n_outputs):
            axis.step(plot_range[self.init_len:], self.ref[self.init_len:,
                     i, :], label=r"$ref_{}$".format(i), **pltargs)

        axis.fill_betweenx(np.arange(-5, 5, 0.1), 0, self.init_len*self.model.Ts,
                          alpha=0.4, label="Init stage", color="#7F7F7F")

        axis.legend()

    def plot_loss(self, axis = None, **pltargs) -> None:
        pltargs.setdefault("linewidth", 1)

        y = self.model.get_y()
        length = y.shape[0] - self.init_len
        plot_range = np.linspace(
            self.init_len*self.model.Ts, length*self.model.Ts, length, endpoint=False)

        axis.semilogy(plot_range, np.array(self.objective)[:, 0, 0], label="$f$")
        axis.set_xlim(0,  y.shape[0]*self.model.Ts)
        axis.legend()
