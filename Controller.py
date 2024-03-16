import numpy as np
from enum import Enum
from scipy import linalg
import matplotlib.pyplot as plt
from typing import Callable, List
from dataclasses import dataclass, field

import casadi as cs
from casadi.tools import *

from SysModels import System, LinearSystem, NonlinearSystem
from helper import hankelize, pagerize, RndSetpoint, Bound
from rcracers.utils.geometry import Polyhedron, Ellipsoid


class ControllerType(Enum):
    VANILLA = 0
    PREDICTIVE = 1


class SMStruct(Enum):
    HANKEL = 0
    PARTIAL_HANKEL = 1
    PAGE = 2


class OCPType(Enum):
    CANONICAL = 0
    REGULARIZED = 1


@dataclass
class InvSetResults:
    n_iter: int = 0
    iterations: List[Polyhedron] = field(default_factory=list)
    success: bool = False

    def increment(self):
        self.n_iter += 1

    @property
    def solution(self) -> Polyhedron:
        return self.iterations[-1]


class Controller:
    controller_type: ControllerType = None

    def __init__(self, model: System, **kwargs) -> None:
        self.model = model
        self.closed_loop = None
        self.u = None

    def __call__(self, x: np.ndarray, r: np.ndarray) -> np.ndarray:
        raise NotImplementedError()


class OpenLoop(Controller):
    controller_type = ControllerType.VANILLA

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
        inst.__rnd_input_seq(length=length,
                             lbu=model.input_constraint.lb,
                             ubu=model.input_constraint.ub)
        return inst

    def __set_input_sequence(self, u: np.ndarray) -> None:
        self.u = u

    def __rnd_input_seq(self, length: int, lbu: np.ndarray, ubu: np.ndarray, switch_prob: float = 0.6) -> None:
        assert (lbu.shape == ubu.shape)

        self.u = np.ones([length, lbu.shape[0], ubu.shape[1]])

        for k in range(length):
            if np.random.rand() <= switch_prob:
                self.u[k] = np.random.uniform(-400, 400, [1, 1])
            else:
                self.u[k] = self.u[k-1]

    def __call__(self, x: np.ndarray, k: int) -> np.ndarray:
        u = self.u[0]
        self.u = np.roll(self.u, -1)
        return u, None


class LQRController(Controller):
    controller_type = ControllerType.VANILLA

    def __init__(self, model: LinearSystem, **kwargs) -> None:
        super().__init__(model)

        self.closed_loop = True
        self.K = None

        try:
            self.Q = kwargs["Q"]
        except KeyError:
            C = model.C
            ns = model.n
            self.Q = C.T@C + np.eye(ns, ns) * 1e-2

        try:
            self.R = kwargs["R"]
        except KeyError:
            nu = model.m
            self.R = np.eye(nu, nu) * 0.1

        self.compute_K()

    def set_weights(self, Q: np.ndarray, R: np.ndarray) -> None:
        self.Q, self.R = Q, R
        self.compute_K()

    def compute_K(self) -> None:
        A = self.model.A
        B = self.model.B

        P = linalg.solve_discrete_are(A, B, self.Q, self.R)
        self.K = -np.linalg.inv(self.R + B.T@P@B)@B.T@P@A

    def __call__(self, x: np.ndarray, r: int) -> np.ndarray:
        u = self.K@(x - r)

        if u >= self.model.input_constraint.ub:
            u = self.model.input_constraint.ub
        elif u <= self.model.input_constraint.lb:
            u = self.model.input_constraint.lb

        return u, None


class MPC(Controller):
    controller_type = ControllerType.PREDICTIVE

    def __init__(self, model: System,
                 horizon: int,
                 *,
                 enforce_term: bool = False,
                 **kwargs) -> None:

        super().__init__(model)

        kwargs.setdefault("output_bound", self.model.output_constraint)
        kwargs.setdefault("input_bound", self.model.input_constraint)
        kwargs.setdefault("Q", np.eye(self.model.p, self.model.p) * 10)
        kwargs.setdefault("R", np.eye(self.model.m, self.model.m) * 1e-2)
        kwargs.setdefault("Pf", np.eye(self.model.p, self.model.p) * 10)

        self.closed_loop = True

        self.horizon = horizon

        self.Q = kwargs["Q"]
        self.R = kwargs["R"]
        self.Pf = kwargs["Pf"]

        self.enforce_term = enforce_term
        self.Yf = self.__compute_invariant_set()

        self.problem = None
        self.solver = None

        self.state_bound: Bound = kwargs["output_bound"]
        self.input_bound: Bound = kwargs["input_bound"]

        self.opt_params: struct_symMX = None
        self.opt_vars: struct_symMX = None
        self.param_vals = None
        self.var_vals = None
        self.ref = None    # Tracking reference

        self.solver, self.bounds = self.__build_mpc_problem()

        self.objective = []

    def __build_mpc_problem(self) -> None:

        x0 = cs.SX.sym(f"x0", (self.model.n, 1))
        x = x0

        lb_states = self.state_bound.lb
        ub_states = self.state_bound.ub
        lb_inputs = self.input_bound.lb
        ub_inputs = self.input_bound.ub

        state_constraints = []
        lbu, ubu, lbx, ubx = [], [], [], []
        cost = 0

        Q, R = self.Q, self.R

        opt_params = struct_symMX([entry('x0', shape=(self.model.n, 1)),
                                   entry('ref', shape=(self.model.n, 1))])

        opt_vars = struct_symMX([entry("u", shape=(self.model.m), repeat=self.horizon)])

        self.param_vals = opt_params(0)
        self.var_vals = opt_vars(0)

        x = opt_params["x0"]

        for k in range(self.horizon):
            uk = opt_vars["u", k]
            wk = np.zeros((self.model.m2, 1))
            
            y = self.model._output(x, uk)

            cost += y.T@Q@y + uk.T@R@uk

            x = self.model._f(x, uk, wk)
        
            state_constraints.append(y)
            lbu.append(lb_inputs)
            ubu.append(ub_inputs)
            lbx.append(lb_states)
            ubx.append(ub_states)

        y = self.model._output(x, np.zeros((self.model.m, 1)))
        cost += y.T@(10*Q)@y

        if self.enforce_term and isinstance(self.Yf, Polyhedron):
            state_constraints.append(self.Yf.H @ x - self.Yf.h)
            lbx.append(-np.ones_like(self.Yf.h) * np.infty)
            ubx.append(np.zeros_like(self.Yf.h))
        elif self.enforce_term and isinstance(self.Yf, Ellipsoid):
            ...

        self.problem = {"x": opt_vars,  # optimized variables: input u
                        "f": cost,
                        "g": cs.vertcat(*state_constraints),
                        "p": opt_params}

        opts = {"ipopt.tol": 1e-12,
                "ipopt.max_iter": 200,
                "ipopt.print_level": 0,
                "expand": True,
                "verbose": False,
                "print_time": False}

        solver = nlpsol("solver", "ipopt", self.problem, opts)

        bounds = dict(
            lbx=cs.vertcat(*lbu), ubx=cs.vertcat(*ubu),
            lbg=cs.vertcat(*lbx), ubg=cs.vertcat(*ubx)
        )

        return solver, bounds

    def __build_feasible_output_set(self, K) -> Polyhedron:

        C, D = self.model.C, self.model.D

        H = np.vstack([np.eye(self.model.p),
                       -np.eye(self.model.p)]) @ (C + D@K)

        h = np.vstack([self.model.output_constraint.ub,
                       -self.model.output_constraint.lb])

        return Polyhedron.from_inequalities(H, h)

    def __build_feasible_input_set(self, K: np.ndarray) -> Polyhedron:
        H = np.vstack([np.eye(self.model.m),
                       -np.eye(self.model.m)]) @ K

        h = np.vstack([self.model.input_constraint.ub,
                       -self.model.input_constraint.lb])

        return Polyhedron.from_inequalities(H, h)

    def __compute_pre(self, set: Polyhedron, Acl: np.ndarray):
        return set.from_inequalities(set.H@Acl, set.h)

    def __invariant_iter(self, Ω_0: Polyhedron, pre: Callable, max_iter: int):
        result = InvSetResults()
        result.iterations.append(Ω_0)

        Ω = Ω_0
        while result.n_iter <= max_iter:
            Ω_next = pre(Ω).intersect(Ω).canonicalize()
            result.iterations.append(Ω_next)
            if Ω_next == Ω:
                result.success = True
                break
            Ω = Ω_next
            result.increment()

        return result

    def __compute_invariant_set(self, max_iter: int = 200):

        if isinstance(self.model, LinearSystem):
            A, B = self.model.A, self.model.B
        elif isinstance(self.model, NonlinearSystem):
            ...

        Q = np.eye(self.model.n) * 3e2

        # find the LQR solution of the problem
        P = linalg.solve_discrete_are(A, B, Q, self.R)
        K = -np.linalg.inv(self.R + B.T@P@B)@B.T@P@A

        Xu = self.__build_feasible_input_set(K)
        Xy = self.__build_feasible_output_set(K)

        X_init = Xy.intersect(Xu)
        Ω_0 = X_init

        Acl = A + B@K

        def pre(Ω):
            return self.__compute_pre(Ω, Acl)

        result = self.__invariant_iter(Ω_0, pre, max_iter)

        return result.solution

    def __update_ref(self, r: np.ndarray) -> None:
        self.ref = np.concatenate(
            [self.ref, np.atleast_3d(r.squeeze())], axis=0)

    def __call__(self, x: np.ndarray, r: np.ndarray) -> np.ndarray:
        self.param_vals['x0'] = x
        self.param_vals['ref'] = vertsplit(r.squeeze())
        # self.__update_ref(r)

        res = self.solver(p=self.param_vals, **self.bounds)

        # Extract optimal solution
        self.var_vals.master = res['x']
        loss_val = res['f']
        opti_u = self.var_vals['u']

        self.objective.append(loss_val)

        return opti_u[0], opti_u[self.model.m:]


class DeePC(Controller):
    controller_type = ControllerType.PREDICTIVE

    def __init__(self, model: System, T_ini: int, horizon: int,
                 init_law: Controller, init_len: int = None, data_mat: SMStruct = SMStruct.HANKEL, **kwargs) -> None:
        super().__init__(model)

        kwargs.setdefault("output_bound", self.model.output_constraint)
        kwargs.setdefault("input_bound", self.model.input_constraint)

        kwargs.setdefault("λ_s", 0)
        kwargs.setdefault("λ_g", 0)

        kwargs.setdefault("Q", np.eye(self.model.p, self.model.p) * 10)
        kwargs.setdefault("R", np.eye(self.model.m, self.model.m) * 1e-2)
        kwargs.setdefault("Pf", np.eye(self.model.p, self.model.p) * 10)

        self.closed_loop = True

        self.T_ini = T_ini
        self.init_len = init_len
        self.horizon = horizon
        self.init_ctrl = init_law
        self.sm_struct = data_mat

        self.Q = kwargs["Q"]
        self.R = kwargs["R"]
        self.Pf = kwargs["Pf"]
        self.λ_s = kwargs["λ_s"]
        self.λ_g = kwargs["λ_g"]

        self.output_bound: Bound = kwargs["output_bound"]
        self.input_bound: Bound = kwargs["input_bound"]

        self.problem = None
        self.solver = None
        self.opt_p = None  # Trajectory constraint: RHS()
        self.opti_vars = None
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
                self.init_len = 2*L*((nu+1)*(nx+1)-1)

        self.ref = np.zeros([self.init_len, ny, 1])
        if self.init_ctrl.closed_loop:
            cloop_sp = RndSetpoint(ny, self.init_len, 0,
                                   np.array([[[-1], [1]]]))
            self.ref = cloop_sp()

        self.model.simulate(
            self.init_len, control_law=self.init_ctrl, reference=None)

        self.__set_constraints()
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
                "print_time": False}

        self.solver = nlpsol("solver", "ipopt", self.problem, opts)

    def __set_constraints(self) -> None:

        L = self.T_ini + self.horizon

        if self.sm_struct == SMStruct.HANKEL:
            M_u = hankelize(self.model.get_u(), L)
            M_y = hankelize(self.model.get_y(), L)
        elif self.sm_struct == SMStruct.PAGE:
            M_u = pagerize(self.model.get_u(), L, L//2-2)
            M_y = pagerize(self.model.get_y(), L, L//2-2)

        # Split Hu and Hy into Hp and Hf (ETH paper Eq.5)
        U_p, U_f = np.split(M_u, [self.model.m * self.T_ini], axis=0)
        Y_p, Y_f = np.split(M_y, [self.model.p * self.T_ini], axis=0)
        self.Y_f = Y_f
        self.U_f = U_f

        self.opt_p = struct_symMX([entry('u_ini', shape=(self.model.m), repeat=self.T_ini),
                                   entry('y_ini', shape=(self.model.p), repeat=self.T_ini),
                                   entry('ref', shape=(self.model.p))])
        self.opti_vars = struct_symMX([entry("u", shape=(self.model.m), repeat=self.horizon),
                                       entry("y", shape=(self.model.p), repeat=self.horizon),
                                       entry("g", shape=[U_f.shape[1]])])
        self.opti_vars_num = self.opti_vars(0)
        self.opt_p_num = self.opt_p(0)

        if isinstance(self.model, LinearSystem) and not self.model.noisy:
            A = vertcat(U_p, Y_p, U_f, Y_f)

            # U, s, V = np.linalg.svd(A)

            # U[:, 10:] = 0
            # s[10:] = 0
            # V[10:, :] = 0

            # import scipy.linalg

            # A = U@scipy.linalg.diagsvd(s, *A.shape)@V

            b = vertcat(*self.opt_p['u_ini'],
                        *self.opt_p['y_ini'],
                        *self.opti_vars['u'],
                        *self.opti_vars['y'])
        else:
            A = vertcat(U_p, Y_p, U_f)

            U, S, V = np.linalg.svd(A)
            U = U[:, :9] 
            V = V[:9, :] 
            S = np.diag(S[:9])
            A = U@S@V

            b = vertcat(*self.opt_p['u_ini'],
                        *self.opt_p['y_ini'],
                        *self.opti_vars['u'])

        g = self.opti_vars['g']
        self.traj_constraint = A@g - b

        self.A = A

        # input constraints and output constraints
        optim_var = self.opti_vars

        self.lbx = optim_var(-np.inf)
        self.ubx = optim_var(np.inf)

        self.lbx['u'], self.ubx['u'] = self.input_bound.lb, self.input_bound.ub
        self.lbx['y'], self.ubx['y'] = self.output_bound.lb, self.output_bound.ub

    def loss(self) -> cs.MX:
        loss = 0
        Q, R, = self.Q, self.R

        for k in range(self.horizon - 1):
            y_k = self.opti_vars["y", k]
            u_k = self.opti_vars["u", k]

            loss += sum1(y_k.T @ Q @ y_k) + sum1(u_k.T @ R @ u_k)
        
        y_N = self.opti_vars["y", self.horizon-1]
        u_N = self.opti_vars["u", self.horizon-1]
        loss += sum1(y_N.T @ (10*Q) @ y_N) + sum1(u_N.T @ (10*R) @ u_N)

        if self.model.noisy or not isinstance(self.model, LinearSystem):
            # regularization terms
            print("Add regularization")
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

        y_ini, u_ini = self.__build_ini_traj()

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

    def plot_loss(self, axis=None, **pltargs) -> None:
        pltargs.setdefault("linewidth", 1)

        y = self.model.get_y()
        length = y.shape[0] - self.init_len
        plot_range = np.linspace(
            self.init_len*self.model.Ts, length*self.model.Ts, length, endpoint=False)

        axis.semilogy(plot_range, np.array(self.objective)[:, 0, 0], label="$f$")
        axis.set_xlim(0,  y.shape[0]*self.model.Ts)
        axis.legend()
