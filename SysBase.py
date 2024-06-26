from typing import Callable, Any
from dataclasses import dataclass
from rich.progress import track
import numpy as np
import casadi as cs
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from helper import forward_euler, zoh, rk4, Bound


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
        self.n: int = None  # states
        self.m: int = None  # inputs
        self.m2: int = None  # inputs: disturbance
        self.p: int = None  # outputs

        # Simulation
        self.n_steps: int = 0

        # noise
        self.noisy: bool = None
        self.w: float = None  # process noise
        self.v: float = None  # measurement noise

        # constraints
        self.state_constraint: Bound = Bound()
        self.output_constraint: Bound = Bound()
        self.input_constraint: Bound = Bound()

        # Naming
        self.input_names: list = None
        self.state_names: list = None
        self.output_names: list = None

    def __repr__(self) -> str:
        raise NotImplementedError()

    def _init_constraints(self, xb: Bound = None, ub: Bound = None, yb: Bound = None):
        self.state_constraint.ub = np.ones((self.n, 1)) * np.infty
        self.state_constraint.lb = -np.ones((self.n, 1)) * np.infty
        self.output_constraint.ub = np.ones((self.p, 1)) * np.infty
        self.output_constraint.lb = -np.ones((self.p, 1)) * np.infty
        self.input_constraint.ub = np.ones((self.m, 1)) * np.infty
        self.input_constraint.lb = -np.ones((self.m, 1)) * np.infty

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
        return np.random.multivariate_normal(mean, self.w, size=[1]).T

    def _measurement_noise(self) -> np.ndarray:
        mean = np.zeros(self.p)
        return np.random.multivariate_normal(mean, self.v, size=[1]).T

    def _f(self, x0: np.ndarray, u: np.ndarray, w: np.ndarray) -> np.ndarray:
        """

        ## Abstract method to be overridden

        Update the system state:

        * x0: Current state
        * p:  Control input
        * w:  External disturbance

        return: New system state

        """
        raise NotImplementedError()

    def _output(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        # Abstract method to be overridden
        raise NotImplementedError()

    def simulate(self,
                 n_steps: int,
                 *,
                 control_law: Callable = None,
                 observer: Callable = None,
                 reference: Callable = None,
                 disturbance: np.ndarray = None) -> None:

        self.n_steps = n_steps

        if control_law is None:
            def zero_input(x, u):
                return np.zeros((self.m, 1)), None
            control_law = zero_input

        if observer is None:
            def full_state_sensor(xk):
                return self.__x[-1]
            observer = full_state_sensor

        if reference is None:
            def zero_ref():
                return np.zeros([self.p, 1])
            reference = zero_ref

        if disturbance is None:
            disturbance = np.zeros([n_steps, self.m2, 1])

        for k in track(range(n_steps), description="Simulation ...", total=n_steps):
            x_hat = observer(self.__y[-1])
            uk, u_pred = control_law(x_hat, reference())
            x_next = self._f(x0=self.__x[-1], p=uk, w=disturbance[k]) + self._process_noise()
            yk = self._output(x_next, uk) + self._measurement_noise()

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

        self.n_steps: int = 0

        self._set_initial_states(x0)

    def __plot_prediction(self, k: int, **pltargs):
        pltargs.setdefault('color', "#4F4F4F")
        pltargs.setdefault('linewidth', 1.5)
        pltargs.setdefault('linestyle', '--')
        axis = pltargs["plot_use"]

        start = (k + 35) * self.Ts
        end = start + self.__pred_y.shape[0] * self.Ts
        plot_range = np.linspace(
            start, end, self.__pred_y.shape[0], endpoint=False)

        for i in range(self.p):
            axis.plot(plot_range, self.__pred_y[:, i, :], **pltargs)

    def plot_trajectory(self,
                        *,
                        axis: plt.Axes,
                        states: list,
                        trim_exci: bool = False,
                        label_prefix: str = "",
                        **pltargs):
        pltargs.setdefault('linewidth', 1.5)

        if states == Any or states == None:
            states = range(self.p)

        y = self.get_y()
        if trim_exci:
            y = y[-self.n_steps:, :, :]

        plot_range = np.linspace(
            0, y.shape[0]*self.Ts, y.shape[0], endpoint=False)

        for i in states:
            if self.output_names is not None:
                lbl = self.output_names[i] + ": " + label_prefix
            else:
                lbl = r"$y_{}$".format(i) + label_prefix

            axis.step(plot_range, y[:, i, :],
                      label=lbl, **pltargs)

        axis.legend(loc="upper right")
        # axis.hlines(0, xmin=0, xmax=10)
        axis.set_xlabel(r"{Time(s)}")

    def plot_phasespace(self,
                        axis: plt.Axes,
                        *,
                        states: list,
                        trim_exci: bool = False,
                        colormap: np.ndarray = None,
                        **pltargs):
        pltargs.setdefault('linewidth', 3)
        axis.set_aspect("equal", adjustable="box")

        sim_t = self.n_steps * self.Ts

        y = self.get_y()
        t = np.linspace(0, sim_t, self.n_steps)
        if trim_exci:
            y = y[-self.n_steps:, :, :]

        if states == Any or states == None:
            # plot the first two states by default
            states = [0, 1]

        points = np.array([y[:, states[0], :], y[:, states[1], :]]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        if colormap is None:
            cm = t
        else:
            cm = colormap
        
        norm = plt.Normalize(np.min(cm), np.max(cm))

        lc = LineCollection(segments, cmap='coolwarm', norm=norm)
        lc.set_array(cm)
        lc.set_linewidth(2.5)

        line = axis.add_collection(lc)

        axis.margins(0.1, 0.1)

        plt.colorbar(line, ax=axis, location="bottom",
                     shrink = 1.0, label=r"$v_x$")

        return y, t

    def plot_control_input(self,
                           *,
                           axis: plt.Axes,
                           trim_exci: bool = False,
                           label_prefix: str = "",
                           **pltargs):
        pltargs.setdefault("linewidth", 0.7)

        u = self.get_u()
        if trim_exci:
            u = u[-self.n_steps:, :, :]

        plot_range = np.linspace(
            0, u.shape[0]*self.Ts, u.shape[0], endpoint=False)

        for i in range(self.m):
            if self.input_names is not None:
                lbl = self.input_names[i] + ": " + label_prefix
            else:
                lbl = r"$y_{}$".format(i) + label_prefix

            axis.step(plot_range, u[:, i, :],
                      label=lbl, **pltargs)

        axis.legend()
        axis.set_xlabel(r"Time(s)")


class NonlinearSystem(System):
    def __init__(self, states: cs.MX, inputs: cs.MX, outputs: cs.MX,
                 x0: np.ndarray, C: np.ndarray = None, *, w: cs.MX = None, **kwargs) -> None:
        super().__init__(**kwargs)

        self.n = states.shape[0]
        self.m = inputs.shape[0]
        self.m2 = w.shape[0]
        self.p = outputs.shape[0]
        self.C = C if C is not None else np.eye(self.n)  # output matrix

        # self.__dynamics = self._dynamics_sym()
        # self.__F = self.__discrete_dynamics()
        self.__F = rk4(self._dynamics_num, self.Ts)

        self._set_noise(**kwargs)
        self._set_initial_states(x0=x0)
        self._init_constraints()

    def __repr__(self) -> str:
        info = "Nonlinear system"
        return info

    def _dynamics_num(self, x, u, w) -> cs.SX:
        # Abstract method to be overrided
        raise NotImplementedError()

    def _f(self, x0, p, w) -> np.ndarray:
        x_next = self.__F(x0=x0, p=p, w=w)
        # x_next = x_next["xf"]
        # return x_next.full()
        return x_next

    def _output(self, x, u) -> np.ndarray:
        return self.C @ x


class LinearSystem(System):
    def __init__(self, A: np.ndarray, B1: np.ndarray, B2: np.ndarray,
                 C: np.ndarray, D: np.ndarray,
                 x0: np.ndarray, **kwargs) -> None:

        super().__init__(**kwargs)

        B_aug = np.concatenate([B1, B2], axis=1)

        self.A, B_aug = zoh(A, B_aug, self.Ts)
        self.B = np.reshape(B_aug[:, 0], (-1, 1))
        self.B2 = np.reshape(B_aug[:, 1], (-1, 1))
        self.C, self.D = C, D

        self.n = A.shape[1]
        self.m = B1.shape[1]
        self.m2 = B2.shape[1]
        self.p = C.shape[0]

        self._f = self._dynamics

        self._set_noise(**kwargs)
        self._set_initial_states(x0=x0)
        self._init_constraints()

    def __repr__(self) -> str:
        info = "Linear system"
        return info

    def _dynamics(self, x0: np.ndarray, p: np.ndarray, w: np.ndarray = None) -> np.ndarray:

        assert x0.shape == (self.n, 1), "Current state vector ∈ {}".format(x0.shape)  # sanity check
        assert p.shape == (self.m, 1), "Control vector ∈ {}".format(p.shape)  # sanity check
        assert w.shape == (self.m2, 1), "Disturbance vector ∈ {}".format(w.shape)  # sanity check

        x_next = self.A@x0 + self.B@p + self.B2@w

        assert x_next.shape == (self.n, 1), "New state vector ∈ {}".format(x_next.shape)  # sanity check

        return x_next

    def _output(self, x, u) -> np.ndarray:

        y = self.C @ x + self.D @ u
        assert y.shape == (self.p, 1)

        return y

    def ctrl(self):
        C = self.B

        for ord in range(1, self.n):
            C_block = np.linalg.matrix_power(self.A, ord) @ self.B
            C = np.concatenate([C, C_block], axis=1)

        return C

    def obsv(self):
        O = self.C

        for ord in range(1, self.n):
            O_block = self.C @ np.linalg.matrix_power(self.A, ord)
            O = np.concatenate([O, O_block], axis=0)

        return O

    def lag(self):
        O = self.C

        for ord in range(1, self.n):
            O_block = self.C @ np.linalg.matrix_power(self.A, ord)
            O = np.concatenate([O, O_block], axis=0)
            if np.linalg.matrix_rank(O) == self.n:
                break

        return ord
