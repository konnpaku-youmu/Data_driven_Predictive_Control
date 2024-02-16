from typing import Callable
from dataclasses import dataclass
import numpy as np
import casadi as cs
import matplotlib.pyplot as plt
from helper import forward_euler, zoh, runge_kutta


@dataclass
class Constraint:
    lb: np.ndarray = None
    ub: np.ndarray = None


class System:
    def __init__(self, **kwargs) -> None:

        # set default values of keyword arguments
        kwargs.setdefault("Ts", 0.05)

        # System variables
        self.__x: np.ndarray = None
        self.__y: np.ndarray = None
        self.__u: np.ndarray = None
        self.__pred_x: np.ndarray = None
        self.__pred_y: np.ndarray = None
        self.Ts = kwargs["Ts"]

        # Integer invariants
        self.n = None  # states
        self.m = None  # inputs
        self.m2 = None  # inputs: disturbance
        self.p = None  # outputs

        # noise
        self.noisy = None
        self.w = None  # process noise
        self.v = None  # measurement noise

        # plotting
        self.__plot_axis = kwargs["plot_use"]

    def __repr__(self) -> str:
        raise NotImplementedError()

    def _set_initial_states(self, x0: np.ndarray) -> None:
        assert x0.shape[0] == self.n

        self.__x = np.ndarray([1, self.n, 1])
        self.__y = np.ndarray([1, self.p, 1])
        self.__u = np.ndarray([1, self.m, 1])

        self.__u[0] = np.zeros([self.m, 1])
        self.__x[0] = x0
        self.__y[0] = self._output(self.__x[0], self.__u[0])

    def _set_noise(self, **kwargs) -> None:
        kwargs.setdefault("noisy", False)
        kwargs.setdefault("σ_x", np.zeros([self.n, self.n]))
        kwargs.setdefault("σ_y", np.zeros([self.p, self.p]))

        self.noisy = kwargs["noisy"]
        self.w = kwargs["σ_x"] if self.noisy else np.zeros(
            [self.n, self.n])
        self.v = kwargs["σ_y"] if self.noisy else np.zeros(
            [self.p, self.p])

    def __update_u(self, uk: np.ndarray | cs.DM) -> None:
        # compatibility with casadi::MX
        u_next = uk if type(uk) is np.ndarray else uk.full()
        # append u_k
        self.__u = np.concatenate([self.__u, np.atleast_3d(u_next.squeeze())], axis=0)
        return

    def __update_x(self, xk: np.ndarray | cs.DM) -> None:
        # compatibility with casadi::MX
        x_next = xk if type(xk) is np.ndarray else xk.full()
        # append x_k
        self.__x = np.concatenate([self.__x, np.atleast_3d(x_next.squeeze())], axis=0)

    def __update_y(self, yk: np.ndarray | cs.DM) -> None:
        # compatibility with casadi::MX
        y_next = yk if type(yk) is np.ndarray else yk.full()
        # append y_k
        self.__y = np.concatenate([self.__y, np.atleast_3d(y_next.squeeze())], axis=0)

    def __initialize_predictions(self) -> None:
        self.__pred_x = np.ndarray([1, self.n, 1])
        self.__pred_y = np.ndarray([1, self.p, 1])

        self.__pred_x[0] = self.__x[-1]
        self.__pred_y[0] = self._output(self.__pred_x[0], self.__u[-1])

    def __update_x_pred(self, x_pred_k: np.ndarray) -> None:
        x_pred_next = x_pred_k if type(
            x_pred_k) is np.ndarray else x_pred_k.full()
        self.__pred_x = np.concatenate(
            [self.__pred_x, np.atleast_3d(x_pred_next.squeeze())], axis=0)

    def __update_y_pred(self, y_pred_k: np.ndarray) -> None:
        y_pred_next = y_pred_k if type(
            y_pred_k) is np.ndarray else y_pred_k.full()
        self.__pred_y = np.concatenate(
            [self.__pred_y, np.atleast_3d(y_pred_next.squeeze())], axis=0)

    def _process_noise(self) -> np.ndarray:
        mean = np.zeros(self.n)
        return np.multivariate_normal(mean, self.w, size=[1]).T

    def _measurement_noise(self) -> np.ndarray:
        mean = np.zeros(self.p)
        return np.random.multivariate_normal(mean, self.v, size=[1]).T

    def _f(self, x0: np.ndarray, u: np.ndarray, w: np.ndarray) -> np.ndarray:
        # Abstract method to be overrided
        raise NotImplementedError()

    def _output(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        # Abstract method to be overrided
        raise NotImplementedError()

    def simulate(self,
                 n_steps: int,
                 *,
                 control_law: Callable = None,
                 measurement: Callable = None,
                 reference: np.ndarray = None,
                 disturbance: np.ndarray = None) -> None:

        if control_law is None:
            def zero_input(x, u):
                return np.zeros((self.m, 1)), None
            control_law = zero_input

        if measurement is None:
            def full_state_sensor(xk, uk):
                return xk
            output = full_state_sensor
        else:
            output = self._output

        if reference is None:
            reference = np.zeros([n_steps, self.p, 1])

        if disturbance is None:
            disturbance = np.zeros([n_steps, self.m2, 1])

        for k in range(n_steps):
            # TODO: Measurement
            uk, u_pred = control_law(self.__x[-1], reference[k])
            x_next = self._f(x0=self.__x[-1], p=uk, w=disturbance[[k]])
            yk = self._output(x_next, uk)

            # Time update
            self.__update_x(x_next)
            self.__update_u(uk)
            self.__update_y(yk)

    def __prediction_openloop(self, u_pred: np.ndarray) -> None:
        if u_pred is not None:
            for i in range(u_pred.shape[0]):
                x_pred = self._f(x0=self.__pred_x[-1], p=u_pred[i])
                y_pred = self._output(x_pred, u_pred[i])

                self.__update_x_pred(x_pred)
                self.__update_y_pred(y_pred)

    def get_x(self):
        return self.__x[1:]

    def get_y(self):
        return self.__y[1:]

    def get_u(self):
        return self.__u[1:]

    def rst(self, x0: np.ndarray) -> None:
        self.__x = None
        self.__y = None
        self.__u = None

        self._set_initial_states(x0)

    def __plot_prediction(self, k: int, **pltargs):
        pltargs.setdefault('color', "#4F4F4F")
        pltargs.setdefault('linewidth', 1.5)
        pltargs.setdefault('linestyle', '--')
        axis = self.__plot_axis

        start = (k + 35) * self.Ts
        end = start + self.__pred_y.shape[0] * self.Ts
        plot_range = np.linspace(
            start, end, self.__pred_y.shape[0], endpoint=False)

        for i in range(self.p):
            axis.plot(plot_range, self.__pred_y[:, i, :], **pltargs)

    def plot_trajectory(self, **pltargs):
        pltargs.setdefault('linewidth', 1.2)
        axis = self.__plot_axis

        y = self.get_y()

        plot_range = np.linspace(
            0, y.shape[0]*self.Ts, y.shape[0], endpoint=False)

        for i in range(self.p):
            axis.plot(plot_range, y[:, i, :],
                      label=r"$y_{}$".format(i), **pltargs)

        # axis.set_ylim(np.min(self.lb_output), np.max(self.ub_output))
        axis.legend()

    def plot_control_input(self, **pltargs):
        pltargs.setdefault("linewidth", 0.7)
        pltargs.setdefault("linestyle", '--')
        pltargs.setdefault("color", 'g')

        u = self.get_u()

        plot_range = np.linspace(
            0, u.shape[0]*self.Ts, u.shape[0], endpoint=False)

        for i in range(self.m):
            plt.plot(plot_range, u[:, i, :],
                     label=r"$u_{}$".format(i), **pltargs)

        plt.legend()


class NonlinearSystem(System):
    def __init__(self, states: cs.MX, inputs: cs.MX, outputs: cs.MX,
                 x0: np.ndarray, C: np.ndarray = None, **kwargs) -> None:
        super().__init__(**kwargs)

        self.n = states.shape[0]
        self.m = inputs.shape[0]
        self.p = outputs.shape[0]
        self.C = C if C is not None else np.eye(self.n)  # output matrix

        self._sym_x = states
        self._sym_u = inputs
        self._sym_y = outputs

        self.__dynamics = self._define_dynamics()
        self.__F = self.__discrete_dynamics()

        self._set_noise(**kwargs)
        self._set_initial_states(x0=x0)

    def __repr__(self) -> str:
        info = "Nonlinear system"
        return info

    def _define_dynamics(self) -> cs.MX:
        # Abstract method to be overrided
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
                 C: np.ndarray, D: np.ndarray, x0: np.ndarray, B2: np.ndarray = None, **kwargs) -> None:
        super().__init__(**kwargs)

        B_aug = np.concatenate([B, B2], axis=1)

        self.A, B_aug = zoh(A, B_aug, self.Ts)
        self.B = np.reshape(B_aug[:, 1], (-1, 1))
        self.B2 = np.reshape(B_aug[:, 1], (-1, 1))

        self.C, self.D = C, D

        self.n = A.shape[1]
        self.m = B.shape[1]
        self.m2 = B2.shape[1]
        self.p = C.shape[0]

        self._set_noise(**kwargs)
        self._set_initial_states(x0=x0)

    def __repr__(self) -> str:
        info = "Linear system"
        return info

    def _f(self, x0: np.ndarray, p: np.ndarray, w: np.ndarray = None) -> np.ndarray:
        x_next = self.A@x0 + self.B@p + self.B2@w
        return x_next

    def _output(self, x, u) -> np.ndarray:
        y = self.C @ x + self.D @ u + self._measurement_noise()
        return y
