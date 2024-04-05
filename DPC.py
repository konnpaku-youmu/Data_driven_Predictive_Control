import numpy as np
import matplotlib.pyplot as plt

import casadi as cs
from casadi.tools import *

from SysModels import System, LinearSystem
from helper import hankelize, pagerize, Bound, ControllerType, SMStruct

from ControlBase import Controller

class DeePC(Controller):
    controller_type = ControllerType.PREDICTIVE
    name = "DeePC"

    def __init__(self, model: System, T_ini: int, horizon: int,
                 init_law: Controller, init_len: int = None, data_mat: SMStruct = SMStruct.HANKEL, **kwargs) -> None:
        super().__init__(model)

        kwargs.setdefault("output_bound", self.model.output_constraint)
        kwargs.setdefault("input_bound", self.model.input_constraint)

        kwargs.setdefault("λ_s", 0.5)
        kwargs.setdefault("λ_g", 0.5)

        kwargs.setdefault("Q", np.eye(self.model.p, self.model.p) * 10)
        kwargs.setdefault("R", np.eye(self.model.m, self.model.m) * 1e-2)
        kwargs.setdefault("Pf", np.eye(self.model.p, self.model.p) * 10)

        self.closed_loop = True

        self.T_ini = T_ini
        self.init_len = init_len
        self.horizon = horizon
        self.init_ctrl = init_law
        self.sm_struct = data_mat

        self.plot_excitation = False

        self.Q = kwargs["Q"]
        self.R = kwargs["R"]
        self.Pf = kwargs["Pf"]
        self.λ_s = kwargs["λ_s"]
        self.λ_g = kwargs["λ_g"]

        self.output_bound: Bound = kwargs["output_bound"]
        self.input_bound: Bound = kwargs["input_bound"]

        self.problem = None
        self.solver = None
        self.opt_params = None  # Trajectory constraint: RHS()
        self.opt_vars = None
        self.traj_constraint = None
        self.ref = None    # Tracking reference

        self.objective = []

        self.__build_controller(**kwargs)

    def __build_controller(self, **kwargs) -> None:
        L = self.T_ini + self.horizon
        nx = self.model.n
        nu = self.model.m
        ny = self.model.p

        if self.init_len == None:
            if self.sm_struct == SMStruct.HANKEL:
                self.init_len = (nu + 1) * (L + nx) - 1
            elif self.sm_struct == SMStruct.PAGE:
                self.init_len = L*((nu*L+1)*(nx+1)-1)

        self.ref = np.zeros([self.init_len, ny, 1])

        # initial excitation
        self.model.simulate(
            self.init_len, control_law=self.init_ctrl, reference=None)

        if self.plot_excitation:
            self.plot_init_excitation()

        self.__set_constraints()
        cost = self.loss()

        self.problem = {"x": self.opt_vars,
                        "f": cost,
                        "g": self.traj_constraint,
                        "p": self.opt_params}

        opts = {"ipopt.tol": 1e-12,
                "ipopt.max_iter": 200,
                "ipopt.print_level": 0,
                "expand": True,
                "verbose": False,
                "print_time": False}

        self.solver = nlpsol("solver", "ipopt", self.problem, opts)

    def __set_constraints(self) -> None:

        L = self.T_ini + self.horizon

        if self.sm_struct == SMStruct.HANKEL:
            M_u = hankelize(self.model.get_u(), L)
            M_y = hankelize(self.model.get_y(), L)
        elif self.sm_struct == SMStruct.PAGE:
            M_u = pagerize(self.model.get_u(), L, L)
            M_y = pagerize(self.model.get_y(), L, L)

        # Split Hu and Hy into Hp and Hf (ETH paper Eq.5)
        U_p, U_f = np.split(M_u, [self.model.m * self.T_ini], axis=0)
        Y_p, Y_f = np.split(M_y, [self.model.p * self.T_ini], axis=0)

        self.Y_p, self.Y_f = Y_p, Y_f
        self.U_f = U_f

        self.opt_params = struct_symMX([entry('u_ini', shape=(self.model.m), repeat=self.T_ini),
                                        entry('y_ini', shape=(self.model.p), repeat=self.T_ini),
                                        entry('ref', shape=(self.model.p))])

        self.opt_vars = struct_symMX([entry("u", shape=(self.model.m), repeat=self.horizon),
                                      entry("y", shape=(self.model.p), repeat=self.horizon),
                                      entry("g", shape=[U_f.shape[1]])])

        self.var_val = self.opt_vars(0)
        self.param_val = self.opt_params(0)

        if isinstance(self.model, LinearSystem) and not self.model.noisy:

            A = vertcat(U_p,
                        U_f,
                        Y_p,
                        Y_f)

            b = vertcat(*self.opt_params['u_ini'],
                        *self.opt_vars['u'],
                        *self.opt_params['y_ini'],
                        *self.opt_vars['y'])
        else:
            A = vertcat(U_p, U_f, Y_f)

            b = vertcat(*self.opt_params['u_ini'],
                        *self.opt_vars['u'],
                        *self.opt_vars['y'])

        g = self.opt_vars['g']
        self.traj_constraint = A@g - b

        self.data_mat = A

        # input constraints and output constraints
        optim_var = self.opt_vars

        self.lbx = optim_var(-np.inf)
        self.ubx = optim_var(np.inf)

        self.lbx['u'], self.ubx['u'] = self.input_bound.lb, self.input_bound.ub
        self.lbx['y'], self.ubx['y'] = self.output_bound.lb, self.output_bound.ub

        return

    def loss(self) -> cs.MX:
        loss = 0
        Q, R, = self.Q, self.R

        for k in range(self.horizon - 1):
            y_k = self.opt_vars["y", k]
            u_k = self.opt_vars["u", k]

            loss += sum1(y_k.T @ Q @ y_k) + sum1(u_k.T @ R @ u_k)

        y_N = self.opt_vars["y", -1]
        u_N = self.opt_vars["u", -1]
        loss += sum1(y_N.T @ (10*Q) @ y_N) + sum1(u_N.T @ (R) @ u_N)

        if self.model.noisy or not isinstance(self.model, LinearSystem):
            # regularization terms
            print("Add regularization")
            g = self.opt_vars["g"]
            Y_p = self.Y_p
            y_ini = vertcat(*self.opt_params['y_ini'])
            esti_err = Y_p@g - y_ini
            loss += self.λ_s * cs.norm_2(esti_err)**2
            loss += self.λ_g * cs.norm_2(g)**2

        return loss

    def __update_ref(self, r: np.ndarray) -> None:
        self.ref = np.concatenate(
            [self.ref, np.atleast_3d(r.squeeze())], axis=0)
        return

    def __call__(self, x: np.ndarray, r: np.ndarray) -> np.ndarray:

        y_ini, u_ini = self.__build_ini_traj()

        self.param_val['u_ini'] = vertsplit(u_ini)
        self.param_val['y_ini'] = vertsplit(y_ini)
        self.param_val['ref'] = vertsplit(r.squeeze())
        self.__update_ref(r)

        res = self.solver(p=self.param_val, lbg=0, ubg=0,
                          lbx=self.lbx, ubx=self.ubx)

        # Extract optimal solution
        self.var_val.master = res['x']
        loss_val = res['f']
        opti_g = self.var_val['g']

        self.objective.append(loss_val)

        u = self.U_f @ opti_g

        return u[:self.model.m], u[self.model.m:]

    def __build_ini_traj(self):
        y_ini = self.model.get_y()[-self.T_ini:]
        u_ini = self.model.get_u()[-self.T_ini:]

        assert y_ini.shape[0] == u_ini.shape[0], "Input/Output length mismatch"

        if y_ini.shape[0] != self.T_ini:
            p_len = self.T_ini - y_ini.shape[0]
            padding_y = np.zeros([p_len, self.model.p, 1])
            padding_u = np.zeros([p_len, self.model.m, 1])
            y_ini = np.vstack([padding_y,
                               y_ini])
            u_ini = np.vstack([padding_u,
                               u_ini])

        y_ini = y_ini.squeeze()
        u_ini = u_ini.squeeze()

        return y_ini, u_ini

    def get_total_loss(self):
        return np.sum(np.array(self.objective))

    def plot_init_excitation(self) -> None:
        fig = plt.figure(figsize=(12, 4))
        ax0 = fig.add_subplot(1, 2, 1)
        ax1 = fig.add_subplot(1, 2, 2)
        ax2 = plt.twinx(ax1)
        fig.tight_layout()

        ax0.set_title(r"\textbf{Excitation control input}")
        ax0.set_ylabel(r"{Force ($N$)}")
        ax1.set_title(r"\textbf{System output: Excitation phase}")
        ax1.set_ylabel(r"{Displacement ($m$)}")
        ax2.set_ylabel(r"{Acceleration ($\sfrac{m}{s^2}$)}")

        self.model.plot_control_input(axis=ax0)

        self.model.plot_trajectory(axis=ax1, states=[0], color="b",
                                   label_prefix=r"Excitation")
        ax1.legend(loc=3, framealpha=1)

        self.model.plot_trajectory(axis=ax2, states=[1], color="g",
                                   label_prefix=r"Excitation")
        ax2.legend(loc=0, framealpha=1)

        return

    def plot_reference(self, axis=None, **pltargs) -> None:
        pltargs.setdefault('linewidth', 1)

        y = self.model.get_y()
        plot_range = np.linspace(
            0, y.shape[0]*self.model.Ts, y.shape[0], endpoint=False)

        for i in range(self.model.p):
            axis.step(plot_range[self.init_len:],
                      self.ref[self.init_len:, i, :],
                      label=r"$ref_{}$".format(i), **pltargs)

        axis.fill_betweenx(np.arange(-5, 5, 0.1), 0, self.init_len*self.model.Ts,
                           alpha=0.4, label="Init stage", color="#7F7F7F")

        axis.legend()

        return

    def plot_loss(self, axis=None, **pltargs) -> None:
        pltargs.setdefault("linewidth", 1)

        y = self.model.get_y()
        length = y.shape[0]
        plot_range = np.linspace(
            0, length*self.model.Ts, length, endpoint=False)

        axis.semilogy(plot_range, np.array(self.objective)[
                      :, 0, 0], label=r"$f$:{{{0}}}, $N = {{{1}}}$, $T_{{ini}} = {{{2}}}$".format(self.name, self.horizon, self.T_ini))
        axis.set_xlim(0,  y.shape[0]*self.model.Ts)
        axis.legend()

        return

    def plot_data_mat_cov(self, axis=None, **pltargs) -> None:
        Σ = np.cov(self.data_mat)
        fig, axc = plt.subplots(1, 1, figsize=(5, 5))
        fig.tight_layout()

        axc.matshow(Σ, cmap="coolwarm")
        axc.set_xlabel(r"$\Sigma$")

        return

    def plot_data_mat_svd(self, axis=None, **pltargs) -> None:
        U, S, V = np.linalg.svd(self.data_mat)
        fig, [axu, axs, axv] = plt.subplots(1, 3, figsize=(16, 5))
        fig.tight_layout()

        axu.matshow((U), cmap="coolwarm")
        axu.set_xlabel(r"$U$")
        axs.plot(S)
        axs.set_xlabel(r"$S$")
        axv.matshow(V, cmap="coolwarm")
        axv.set_xlabel(r"$V^\mathsf{T}$")

        return