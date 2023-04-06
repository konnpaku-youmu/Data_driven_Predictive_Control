import numpy as np
from typing import Callable
import casadi as cs
import matplotlib.pyplot as plt
from helper import forward_euler, zoh, runge_kutta, SetpointGenerator


class System:
    def __init__(self, **kwargs) -> None:
        # set default values of keyword arguments
        kwargs.setdefault("Ts", 0.05)

        # state variables
        self.__x = None
        self.__y = None
        self.__u = None
        self.Ts = kwargs["Ts"]

        self.n_states = None
        self.n_inputs = None
        self.n_outputs = None

        # noise
        self.noisy = None
        self.__σ_x = None  # process noise
        self.__σ_y = None  # measurement noise

    def _set_initial_states(self, x0: np.ndarray) -> None:
        assert x0.shape[0] == self.n_states

        self.__x = np.ndarray([1, self.n_states, 1])
        self.__y = np.ndarray([1, self.n_outputs, 1])
        self.__u = np.ndarray([1, self.n_inputs, 1])

        self.__u[0] = np.zeros([self.n_inputs, 1])
        self.__x[0] = x0
        self.__y[0] = self._output(self.__x[0], self.__u[0])

    def _set_noise(self, **kwargs) -> None:
        kwargs.setdefault("noisy", False)
        kwargs.setdefault("σ_x", np.zeros([self.n_states, self.n_states]))
        kwargs.setdefault("σ_y", np.zeros([self.n_outputs, self.n_outputs]))

        self.noisy = kwargs["noisy"]
        self.__σ_x = kwargs["σ_x"] if self.noisy else np.zeros(
            [self.n_states, self.n_states])
        self.__σ_y = kwargs["σ_y"] if self.noisy else np.zeros(
            [self.n_outputs, self.n_outputs])

    def rst(self, x0: np.ndarray) -> None:
        self.__x = None
        self.__y = None
        self.__u = None

        self._set_initial_states(x0)

    def __update_u(self, uk: np.ndarray) -> None:
        # compatibility with casadi::MX
        u_next = uk if type(uk) is np.ndarray else uk.full()
        self.__u = np.concatenate(
            [self.__u, np.atleast_3d(u_next.squeeze())], axis=0)

    def __update_x(self, xk: np.ndarray) -> None:
        x_next = xk if type(xk) is np.ndarray else xk.full()
        self.__x = np.concatenate(
            [self.__x, np.atleast_3d(x_next.squeeze())], axis=0)

    def __update_y(self, yk: np.ndarray) -> None:
        y_next = yk if type(yk) is np.ndarray else yk.full()
        self.__y = np.concatenate(
            [self.__y, np.atleast_3d(y_next.squeeze())], axis=0)

    def _measurement_noise(self) -> np.ndarray:
        mean = np.zeros(self.n_outputs)
        return np.random.multivariate_normal(mean, self.__σ_y, size=[1]).T

    def _f(self, x, u) -> np.ndarray:
        raise NotImplementedError()

    def _output(self, x, u) -> np.ndarray:
        raise NotImplementedError()

    def simulate(self, n_steps: int, control_law: Callable = None,
                 ref_traj: np.ndarray = None) -> None:

        if control_law is None:
            def dummy_control(x, k):
                return np.zeros((self.n_inputs, 1))
            control_law = dummy_control

        if ref_traj is None:
            ref_traj = np.zeros([n_steps, self.n_outputs, 1])

        for k in range(n_steps):
            uk = control_law(self.__x[-1], ref_traj[k])
            x_next = self._f(x0=self.__x[-1], p=uk)
            yk = self._output(x_next, uk)

            # Time update
            self.__update_x(x_next)
            self.__update_u(uk)
            self.__update_y(yk)

    def get_x(self):
        return self.__x[1:]

    def get_y(self):
        return self.__y[1:]

    def get_u(self):
        return self.__u[1:]

    def plot_trajectory(self, **pltargs):
        pltargs.setdefault('linewidth', 1.2)

        y = self.get_y()

        plot_range = np.linspace(
            0, y.shape[0]*self.Ts, y.shape[0], endpoint=False)

        for i in range(self.n_outputs):
            plt.plot(plot_range, y[:, i, :],
                     label=r"$y_{}$".format(i), **pltargs)

        plt.legend()

    def plot_control_input(self, **pltargs):
        pltargs.setdefault("linewidth", 0.7)
        pltargs.setdefault("linestyle", '--')
        pltargs.setdefault("color", 'g')

        u = self.get_u()

        plot_range = np.linspace(
            0, u.shape[0]*self.Ts, u.shape[0], endpoint=False)

        for i in range(self.n_inputs):
            plt.plot(plot_range, u[:, i, :],
                     label=r"$u_{}$".format(i), **pltargs)

        plt.legend()


class NonlinearSystem(System):
    def __init__(self, states: cs.MX, inputs: cs.MX, outputs: cs.MX, 
                 x0: np.ndarray, C: np.ndarray = None, **kwargs) -> None:
        super().__init__(**kwargs)

        self.n_states = states.shape[0]
        self.n_inputs = inputs.shape[0]
        self.n_outputs = outputs.shape[0]
        self.C = C if C is not None else np.eye(self.n_states)  # output matrix

        self._sym_x = states
        self._sym_u = inputs
        self._sym_y = outputs

        self.__dynamics = self._define_dynamics()
        self.__F = self.__discrete_dynamics()

        self._set_noise(**kwargs)
        self._set_initial_states(x0=x0)

    def _define_dynamics(self) -> cs.MX:
        raise NotImplementedError()
    
    def __discrete_dynamics(self) -> cs.Function:
        DAE = {"x": self._sym_x,
               "p": self._sym_u,
               "ode": self.__dynamics}
        opts = {"tf": self.Ts}
        return cs.integrator('F', 'cvodes', DAE, opts)

    def _f(self, x0, p) -> np.ndarray:
        x_next = self.__F(x0=x0, p=p)
        x_next = x_next["xf"]
        return x_next.full()

    def _output(self, x, u) -> np.ndarray:
        return self.C @ x + self._measurement_noise()


class LinearSystem(System):
    def __init__(self, A: np.ndarray, B: np.ndarray,
                 C: np.ndarray, D: np.ndarray, x0: np.ndarray, **kwargs) -> None:
        super().__init__(**kwargs)

        self.A, self.B = zoh(A, B, self.Ts)
        self.C, self.D = C, D

        self.n_states = A.shape[1]
        self.n_inputs = B.shape[1]
        self.n_outputs = C.shape[0]

        self._set_noise(**kwargs)
        self._set_initial_states(x0=x0)

    def _f(self, x0, p) -> np.ndarray:
        x_next = self.A@x0 + self.B@p
        return x_next

    def _output(self, x, u) -> np.ndarray:
        y = self.C @ x + self.D @ u + self._measurement_noise()
        return y
