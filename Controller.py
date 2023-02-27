import numpy as np
from typing import Any, Tuple, Callable
from scipy import linalg
import matplotlib.pyplot as plt

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

    def set_input_sequence(self, u:np.ndarray) -> None:
        self.u = u
    
    def generate_rnd_input_seq(self, len:int, lbu:np.ndarray, ubu:np.ndarray, switch_prob: float = 0.05) -> None:
        assert(lbu.shape == ubu.shape)

        self.u = np.zeros([len, lbu.shape[0], ubu.shape[1]])

        for k in range(len):
            if np.random.rand() <= switch_prob:
                self.u[k] = np.random.uniform(lbu, ubu, [lbu.shape[0], ubu.shape[1]])
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

    def __call__(self, x: np.ndarray, k: int) -> np.ndarray:
        return self.K@x


class DeePC(Controller):
    def __init__(self, model: LinearSystem, **kwargs) -> None:
        super().__init__(model)

        self.T_ini = kwargs["T_ini"]
        self.N = kwargs["N"]
        self.init_ctrl = kwargs["init_law"]
        self.exct_bounds = kwargs["excitation_bounds"]

    def build_controller(self, **kwargs) -> None:
        min_exc_len = (self.model.n_inputs + 1) * (self.T_ini + self.N + self.model.n_states) - 1
        
        ## Excite the system
        x0 = np.zeros([self.model.n_states, 1])
        sp_gen = SetpointGenerator(self.model, min_exc_len, 0, "rand", self.exct_bounds)
        self.model.simulate(x0, min_exc_len, control_law=self.init_ctrl, tracking_target=sp_gen())

        H_u = hankelize(self.model.u, self.T_ini+self.N)
        H_y = hankelize(self.model.y, self.T_ini+self.N)

        ## Check full-rank condition
        H_u_pe = hankelize(self.model.u, self.T_ini+self.N+self.model.n_states)
        if np.linalg.matrix_rank(H_u_pe):
            print("Persistently excited of order T_ini+N+n = {}".format(self.T_ini+self.N+self.model.n_states))
        
        ## Split Hu and Hy into Hp and Hf (ETH paper Eq.5)
        U_p, U_f = np.split(H_u, [self.model.n_inputs * self.T_ini], axis=0)
        Y_p, Y_f = np.split(H_y, [self.model.n_output * self.T_ini], axis=0)

        print(U_p.shape, U_f.shape)
        print(Y_p.shape, Y_f.shape)
        
        
    def __call__(self, x: np.ndarray, k: int) -> np.ndarray:
        return super().__call__(x, k)


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
        
        assert(len(trac_states) == len(shapes))
        assert(bounds.shape[2] == len(trac_states))

        for state, shape, bound in zip(trac_states, shapes, bounds):
            sp_state = np.zeros([n_steps, 1])

            if shape == "ramp":
                pass
            elif shape == "step":
                kwargs.setdefault("step_time", self.sim_steps // 3)
                step_time = kwargs["step_time"]

            elif shape == "rand":
                kwargs.setdefault("switching_prob", 0.5)
                switching_prob = kwargs["switching_prob"]

                for k in range(n_steps):
                    if np.random.rand() <= switching_prob:
                        sp_state[k] = np.random.uniform(np.min(bound), np.max(bound))
                    else:
                        sp_state[k] = sp_state[k-1]
                
                self.sp[:, state, :] = sp_state
    
    def plot(self) -> None:
        sim_range = np.linspace(0, self.sim_steps*self.model.Ts, self.sim_steps)
        for i in range(self.model.n_states):
            plt.step(sim_range, self.sp[:, i, :])
    
    def __call__(self) -> np.ndarray:
        return self.sp
