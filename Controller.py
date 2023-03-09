import numpy as np
from typing import Any, Tuple, Callable
from scipy import linalg
import matplotlib.pyplot as plt

import casadi as cs
from casadi.tools import *

from System import LinearSystem
from helper import hankelize


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

        self.T_ini = kwargs["T_ini"]
        self.N = kwargs["N"]
        self.init_ctrl = kwargs["init_law"]
        self.exct_bounds = kwargs["excitation_bounds"]
        
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
        self.ref = np.ones([self.N, self.model.n_output, 1])*np.random.uniform(-0.5, 0.5)    # Tracking reference
        self.traj_constraint = None

    def build_controller(self, **kwargs) -> None:
        min_exc_len = (self.model.n_inputs + 1) * \
            (self.T_ini + self.N + self.model.n_states) - 1
        # plt.vlines(min_exc_len*self.model.Ts, ymin=-1, ymax=1)

        # Excite the system
        x0 = np.zeros([self.model.n_states, 1])
        sp_gen = SetpointGenerator(
            self.model, min_exc_len, 0, "step", self.exct_bounds)
        sp_gen.plot()
        self.model.simulate(
            x0, min_exc_len, control_law=self.init_ctrl, tracking_target=sp_gen())

        self.set_constraints()
        cost = self.loss()

        self.problem = {"x": self.opti_vars,
                        "f": cost,
                        "g": self.traj_constraint,
                        "p": self.opt_p}

        opts = {"ipopt.tol": 1e-12, "ipopt.max_iter":50, "ipopt.print_level": 0, "expand": True, "verbose": False, "print_time":False}
        self.solver = nlpsol("solver", "ipopt", self.problem, opts)

    def set_constraints(self) -> None:
        H_u = hankelize(self.model.u, self.T_ini+self.N)
        H_y = hankelize(self.model.y, self.T_ini+self.N)

        # Check full-rank condition
        H_u_pe = hankelize(self.model.u, self.T_ini+self.N+self.model.n_states)
        if np.linalg.matrix_rank(H_u_pe):
            print("Persistently excited of order T_ini+N+n = {}".format(self.T_ini +
                  self.N+self.model.n_states))

        # Split Hu and Hy into Hp and Hf (ETH paper Eq.5)
        U_p, U_f = np.split(H_u, [self.model.n_inputs * self.T_ini], axis=0)
        Y_p, Y_f = np.split(H_y, [self.model.n_output * self.T_ini], axis=0)
        self.U_f = U_f

        self.opt_p = struct_symMX([entry('u_ini', shape=(self.model.n_inputs), repeat=self.T_ini),
                                   entry('y_ini', shape=(self.model.n_output), repeat=self.T_ini),
                                   entry('ref', shape=(self.model.n_output), repeat=self.N)])
        self.opti_vars = struct_symMX([entry("u", shape=(self.model.n_inputs), repeat=self.N),
                                       entry("y", shape=(self.model.n_output), repeat=self.N),
                                       entry("g", shape=[U_f.shape[1]])])
        self.opti_vars_num = self.opti_vars(0)
        self.opt_p_num = self.opt_p(0)

        A = vertcat(U_p, Y_p, U_f, Y_f)
        b = vertcat(*self.opt_p['u_ini'], *self.opt_p['y_ini'], 
                    *self.opti_vars['u'], *self.opti_vars['y'])
        g = self.opti_vars['g']
        self.traj_constraint = A@g - b

        # input constraints and output constraints
        optim_var = self.opti_vars
        self.lbx = optim_var(-np.inf)
        self.ubx = optim_var(np.inf)
        self.lbx['u'] = -5.0
        self.ubx['u'] = 5.0
        self.lbx['y'] = np.array([[-1], [-0.2]])
        self.ubx['y'] = np.array([[1], [0.2]])

    def loss(self) -> cs.MX:
        loss = 0
        Q, R, = self.Q, self.R
        for k in range(self.N):
            y_k = self.opti_vars["y", k] - self.opt_p['ref', k]
            u_k = self.opti_vars["u", k]
            loss += sum1(y_k.T @ Q @ y_k) + sum1(u_k.T @ R @ u_k)
        return loss

    def __call__(self, x: np.ndarray, r: int) -> np.ndarray:
        y_Tini = self.model.y[-self.T_ini:].squeeze()
        u_Tini = self.model.u[-self.T_ini:].squeeze()
        self.opt_p_num['u_ini'] = vertsplit(u_Tini)
        self.opt_p_num['y_ini'] = vertsplit(y_Tini)
        res = self.solver(p=self.opt_p_num, lbg=0, ubg=0,
                          lbx=self.lbx, ubx=self.ubx)

        # Extract optimal solution
        self.opti_vars_num.master = res['x']
        opti_g = self.opti_vars_num['g']

        u = self.U_f @ opti_g

        return u[0]

class SetpointGenerator:
    def __init__(self, model: LinearSystem, n_steps,
                 trac_states: list, shapes: list,
                 bounds: np.ndarray, **kwargs) -> None:

        self.model = model
        self.sim_steps = n_steps
        self.sp = np.zeros([n_steps, model.n_states, 1])

        if isinstance(trac_states, int) and isinstance(shapes, str):
            trac_states = [trac_states]
            shapes = [shapes]

        assert (len(trac_states) == len(shapes))
        assert (bounds.shape[2] == len(trac_states))

        for state, shape, bound in zip(trac_states, shapes, bounds):
            sp_state = np.zeros([n_steps, 1])

            if shape == "ramp":
                pass
            elif shape == "step":
                kwargs.setdefault("step_time", int(1/self.model.Ts))
                kwargs.setdefault("height", 1)
                step_time = kwargs["step_time"]
                print(step_time)
                height = kwargs["height"]
                sp_state[step_time:] = height
            elif shape == "rand":
                kwargs.setdefault("switching_prob", 0.1)
                switching_prob = kwargs["switching_prob"]

                for k in range(n_steps):
                    if np.random.rand() <= switching_prob:
                        sp_state[k] = np.random.uniform(
                            np.min(bound), np.max(bound))
                    else:
                        sp_state[k] = sp_state[k-1]

            self.sp[:, state, :] = sp_state

    def plot(self, **kwargs) -> None:
        sim_range = np.linspace(
            0, self.sim_steps*self.model.Ts, self.sim_steps, endpoint=False)
        for i in range(self.model.n_states):
            plt.plot(sim_range, self.sp[:, i, :], **kwargs)

    def __call__(self) -> np.ndarray:
        return self.sp
