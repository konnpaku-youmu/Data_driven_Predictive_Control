import numpy as np
from typing import Tuple, Callable
import matplotlib.pyplot as plt

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 14
})


def forward_euler(A, B, Ts) -> Tuple[np.ndarray]:
    n_states = A.shape[1]

    Ad = np.eye(n_states) + Ts * A
    Bd = Ts * B

    return Ad, Bd


class LinearSystem:
    def __init__(self, **kwargs) -> None:
        # state variables
        self.x = None
        self.u = None
        self.Ts = None
        self.sim_steps = None
        self.sim_trange = None

    def build_system_model(self, A: np.ndarray, B: np.ndarray, C: np.ndarray, D: np.ndarray, **kwargs) -> None:

        try:
            Ts = kwargs["Ts"]
            self.A, self.B = forward_euler(A, B, Ts)
            self.Ts = Ts
        except KeyError:
            self.A, self.B = A, B
            print("Initialized as continuous time model")

        self.C, self.D = C, D

        self.n_states = A.shape[1]
        self.n_inputs = B.shape[1]

    def f(self, x, u) -> np.ndarray:
        x_next = self.A@x + self.B@u
        return x_next

    def simulate(self, x0: np.ndarray, n_steps: int,
                 control_law: Callable = None, 
                 tracking_target: np.ndarray = None) -> None:

        if control_law is None:
            def dummy_control(x, k):
                return np.zeros((self.n_inputs, 1))
            control_law = dummy_control
            print("No control input. Autonomous system")

        self.x = np.ndarray([n_steps, self.n_states, 1])
        self.u = np.zeros([n_steps, self.n_inputs, 1])
        self.x[0] = x0
        self.sim_steps = n_steps
        self.sim_trange = np.linspace(0, self.sim_steps*self.Ts, self.sim_steps)

        for k in range(1, n_steps):
            uk = control_law(self.x[k-1] - tracking_target[k], k)
            self.u[k] = uk
            x_next = self.f(self.x[k-1], uk)
            self.x[k] = x_next
    
    def plot_trajectory(self, **pltargs):

        for i in range(self.n_states):
            plt.plot(self.sim_trange, self.x[:, i, :], label=r"$x_{}$".format(i), **pltargs)

        plt.legend()
    
    def plot_control_input(self, **pltargs):
        pltargs.setdefault('linestyle', '--')

        for i in range(self.n_inputs):
            plt.step(self.sim_trange, self.u[:, i, :], label=r"$u_{}$".format(i), **pltargs)
        
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



