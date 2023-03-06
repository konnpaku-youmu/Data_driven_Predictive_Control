import numpy as np
from typing import Tuple, Callable, overload
import matplotlib.pyplot as plt
from helper import forward_euler, zoh

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 14
})


class LinearSystem:
    def __init__(self, **kwargs) -> None:
        # state variables
        self.x = None
        self.y = None
        self.u = None
        self.Ts = None
        self.sim_steps = None
        self.sim_trange = None

    def build_system_model(self, A: np.ndarray, B: np.ndarray, C: np.ndarray, D: np.ndarray, **kwargs) -> None:

        try:
            Ts = kwargs["Ts"]
            self.A, self.B = zoh(A, B, Ts)
            self.Ts = Ts
        except KeyError:
            self.A, self.B = A, B
            print("Initialized as continuous time model")

        self.C, self.D = C, D

        self.n_states = A.shape[1]
        self.n_inputs = B.shape[1]
        self.n_output = C.shape[0]

    def f(self, x, u) -> np.ndarray:
        x_next = self.A@x + self.B@u
        return x_next.squeeze()
    
    def output(self, x, u) -> np.ndarray:
        y = self.C@x + self.D@u
        return y
    
    def rst(self, x0:np.ndarray) -> None:
        self.x[0] = x0
    
    def update_u(self, uk:np.ndarray) -> None:
        self.u = np.concatenate([self.u, np.atleast_3d(uk)], axis=0)

    def update_x(self, xk:np.ndarray) -> None:
        self.x = np.concatenate([self.x, np.atleast_3d(xk)], axis=0)

    def update_y(self, yk:np.ndarray) -> None:
        self.y = np.concatenate([self.y, np.atleast_3d(yk)], axis=0)
    
    def simulate(self, x0: np.ndarray, n_steps: int,
                 control_law: Callable = None,
                 tracking_target: np.ndarray = None) -> None:

        if control_law is None:
            def dummy_control(x, k):
                return np.zeros((self.n_inputs, 1))
            control_law = dummy_control
            print("No control input. Autonomous system")

        if tracking_target is None:
            tracking_target = np.zeros([n_steps, self.n_states, 1])

        if self.x is None and self.u is None:
            _x0 = x0
            self.x = np.ndarray([1, self.n_states, 1])
            self.y = np.ndarray([1, self.n_output, 1])
            self.u = np.zeros([1, self.n_inputs, 1])

            self.u[0] = np.zeros([self.n_inputs, 1])
            self.x[0] = _x0
            self.y[0] = self.output(self.x[0], self.u[0])
        
        self.sim_steps = n_steps
        self.sim_trange = np.linspace(
            0, self.sim_steps*self.Ts, self.sim_steps)

        start = self.x.shape[0]
        for k in range(start, start + n_steps):
            uk = control_law(self.x[k-1] - tracking_target[k-start], k)
            self.update_u(uk)
            x_next = self.f(self.x[k-1], self.u[k])
            self.update_x(x_next)
            yk = self.output(self.x[k-1], self.u[k]).squeeze()
            self.update_y(yk)

    def plot_trajectory(self, **pltargs):
        pltargs.setdefault('linewidth', 1)

        plot_range = np.linspace(0, self.y.shape[0]*self.Ts, self.y.shape[0])

        for i in range(self.n_states-2):
            plt.plot(plot_range, self.y[:, i, :],
                     label=r"$x_{}$".format(i), **pltargs)

        plt.legend()

    def plot_control_input(self, **pltargs):
        pltargs.setdefault('linewidth', 0.7)
        pltargs.setdefault('linestyle', '--')

        plot_range = np.linspace(0, self.y.shape[0]*self.Ts, self.y.shape[0])

        for i in range(self.n_inputs):
            plt.step(plot_range, self.u[:, i, :],
                     label=r"$u_{}$".format(i), **pltargs)

        plt.legend()


class SimpleHarmonic(LinearSystem):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        k = kwargs["k"]
        m = kwargs["mass"]

        A = np.array([[0., 1],
                      [-k/m, 0]])

        B = np.array([[0.],
                      [1/m]])

        C = np.array([1., 0.])

        D = np.array([0.])

        self.build_system_model(A, B, C, D, **kwargs)


class InvertedPendulum(LinearSystem):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        Rm = 2.6
        Km = 0.00767
        Kb = 0.00767
        Kg = 3.7
        M = 0.455
        l = 0.305
        m = 0.210
        r = 0.635e-2
        g = 9.81

        A = np.array([[0, 0, 1, 0],
                      [0, 0, 0, 1],
                      [0, -m * g / M, -(Kg**2 * Km * Kb) / (M * Rm * r**2), 0],
                      [0, (M + m) * g / (M * l), (Kg**2 * Km * Kb) / (M * Rm * r**2 * l), 0]])

        B = np.array([[0],
                      [0],
                      [(Km * Kg) / (M * Rm * r)],
                      [(-Km * Kg) / (r * Rm * M * l)]])

        C = np.array([[1., 0., 0., 0.],
                      [0., 1., 0., 0.]])

        D = np.array([[0.],
                      [0.]])

        self.build_system_model(A, B, C, D, **kwargs)
