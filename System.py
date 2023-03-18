import numpy as np
from typing import Tuple, Callable, overload
import casadi as cs
import matplotlib.pyplot as plt
from helper import forward_euler, zoh, runge_kutta

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 14
})


class System:
    def __init__(self, **kwargs) -> None:
        # state variables
        self.x = None
        self.y = None
        self.u = None
        self.Ts = None

        self.n_states = None
        self.n_inputs = None
        self.n_outputs = None

        # noise
        self.noisy = None
        self.σ_y = None  # std of measurement noise

    def build_system_model(self) -> None:
        ...

    def update_u(self, uk: np.ndarray) -> None:
        self.u = np.concatenate([self.u, np.atleast_3d(uk)], axis=0)

    def update_x(self, xk: np.ndarray) -> None:
        self.x = np.concatenate([self.x, np.atleast_3d(xk)], axis=0)

    def update_y(self, yk: np.ndarray) -> None:
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
            self.x = np.ndarray([1, self.n_states, 1])
            self.y = np.ndarray([1, self.n_outputs, 1])
            self.u = np.zeros([1, self.n_inputs, 1])

            self.u[0] = np.zeros([self.n_inputs, 1])
            self.x[0] = x0
            self.y[0] = self.output(self.x[0], self.u[0])

        for k in range(1, n_steps):
            uk = control_law(self.x[-1], tracking_target[k])
            self.update_u(uk)
            x_next = self.f(x0=self.x[-1], p=self.u[-1])
            self.update_x(x_next)
            yk = self.output(self.x[-1], self.u[-1]).squeeze()
            self.update_y(yk)

    def output(self, x, u):
        ...

    def rst(self) -> None:
        self.x = None
        self.y = None
        self.u = None

    def plot_trajectory(self, **pltargs):
        pltargs.setdefault('linewidth', 1)

        plot_range = np.linspace(
            0, self.y.shape[0]*self.Ts, self.y.shape[0], endpoint=False)

        for i in range(self.n_outputs):
            plt.step(plot_range, self.y[:, i, :],
                     label=r"$y_{}$".format(i), **pltargs)

        plt.legend()

    def plot_control_input(self, **pltargs):
        pltargs.setdefault('linewidth', 0.7)
        pltargs.setdefault('linestyle', '--')

        plot_range = np.linspace(
            0, self.y.shape[0]*self.Ts, self.y.shape[0], endpoint=False)

        for i in range(self.n_inputs):
            plt.step(plot_range, self.u[:, i, :],
                     label=r"$u_{}$".format(i), **pltargs)

        plt.legend()


class NonlinearSystem(System):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.dynamics = None

    def build_system_model(self, states: cs.MX, inputs: cs.MX, outputs: cs.MX, **kwargs) -> None:
        # set default values for arguments
        kwargs.setdefault("noisy", False)

        try:
            self.Ts = kwargs["Ts"]
        except KeyError:
            print("Continuous dynamics")

        self.n_states = states.shape[0]
        self.n_inputs = inputs.shape[0]
        self.n_outputs = outputs.shape[0]
        self.sym_x = states
        self.sym_u = inputs
        self.sym_y = outputs

        self.noisy = kwargs["noisy"]

        self.dynamics = self.define_dynamics()
        self.F = self.discrete_dynamics()

    def discrete_dynamics(self) -> None:
        DAE = {"x": self.sym_x,
               "p": self.sym_u,
               "ode": self.dynamics}
        opts = {"tf": self.Ts}
        return cs.integrator('F', 'cvodes', DAE, opts)

    def define_dynamics(self) -> cs.MX:
        ...

    def f(self, x0, p) -> np.ndarray:
        x_next = self.F(x0=x0, p=p)
        x_next = x_next['xf']
        return x_next.full().squeeze()

    def output(self, x, u) -> np.ndarray:
        return x

    def plot_phasespace(self, **pltargs) -> None:
        plt.scatter(self.x[:, 1, :], self.x[:, 0, :], s=0.1)
        plt.axis("equal")


class LinearSystem(System):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def build_system_model(self, A: np.ndarray, B: np.ndarray, C: np.ndarray, D: np.ndarray, **kwargs) -> None:
        try:
            self.Ts = kwargs["Ts"]
            self.A, self.B = forward_euler(A, B, self.Ts)
        except KeyError:
            self.A, self.B = A, B
            print("Continuous dynamics")

        self.C, self.D = C, D

        self.n_states = A.shape[1]
        self.n_inputs = B.shape[1]
        self.n_outputs = C.shape[0]

        kwargs.setdefault("noisy", False)
        self.noisy = kwargs["noisy"]

        if self.noisy:
            kwargs.setdefault("s_y", np.diag([9e-6, 1e-4]))
            self.σ_y = kwargs["s_y"]
        else:
            self.σ_y = np.diag(np.zeros(self.n_outputs))

    def measurement_noise(self) -> np.ndarray:
        return np.random.multivariate_normal(np.zeros(self.n_outputs), self.σ_y, size=[1]).T

    def f(self, x0, p) -> np.ndarray:
        x_next = self.A@x0 + self.B@p
        return x_next.squeeze()

    def output(self, x, u) -> np.ndarray:
        y = self.C@x + self.D@u + self.measurement_noise()
        return y

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


class VolterraEquations(NonlinearSystem):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        x = cs.MX.sym('x', 2)
        u = cs.MX.sym('u', 1)
        y = cs.MX.sym('y', 2)

        self.build_system_model(x, u, y, **kwargs)

    def define_dynamics(self) -> cs.MX:
        x0_dot = 0.3 * self.sym_x[0] - 0.5 * self.sym_x[0] * self.sym_x[1]
        x1_dot = 0.2 * self.sym_x[0] * self.sym_x[1] - 0.1 * self.sym_x[1]

        return cs.vertcat(x0_dot, x1_dot)


class VanderpolOscillator(NonlinearSystem):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        x = cs.MX.sym('x', 2)
        u = cs.MX.sym('u', 1)
        y = cs.MX.sym('y', 2)

        self.build_system_model(x, u, y, **kwargs)

    def define_dynamics(self) -> cs.MX:
        # system equation
        x0_dot = (3-self.sym_x[1]**2)*self.sym_x[0] - self.sym_x[1]
        x1_dot = self.sym_x[0]

        return cs.vertcat(x0_dot, x1_dot)


class LorenzAttractor(NonlinearSystem):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        x = cs.MX.sym('x', 3)
        u = cs.MX.sym('u', 1)
        y = cs.MX.sym('y', 3)

        self.build_system_model(x, u, y, **kwargs)

    def define_dynamics(self) -> cs.MX:
        # system equation
        x0_dot = 10 * (self.sym_x[1] - self.sym_x[0])
        x1_dot = self.sym_x[0] * (28 - self.sym_x[2]) - self.sym_x[1]
        x2_dot = self.sym_x[0] * self.sym_x[1] - 2.2 * self.sym_x[2]

        return cs.vertcat(x0_dot, x1_dot, x2_dot)


if __name__ == "__main__":
    # model = VanderpolOscillator(Ts=0.05)
    # model = LorenzAttractor(Ts=0.05)
    model = VolterraEquations(Ts=0.05)

    x0 = np.array([[0.1], [0.2]])
    model.simulate(x0, 1000)
    model.plot_phasespace()
    plt.show()
