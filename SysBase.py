import numpy as np
from typing import Callable
import casadi as cs
import matplotlib.pyplot as plt
from helper import forward_euler, zoh, runge_kutta


class System:
    def __init__(self, **kwargs) -> None:
        # set default values of keyword arguments
        kwargs.setdefault("Ts", 0.05)

        # state variables
        self.__x = None
        self.__y = None
        self.__u = None
        self.__pred_x = None
        self.__pred_y = None
        self.Ts = kwargs["Ts"]

        self.n_states = None
        self.n_inputs = None
        self.n_outputs = None

        # noise
        self.noisy = None
        self.__σ_x = None  # process noise
        self.__σ_y = None  # measurement noise
        
        # plotting
        self.__plot_axis = kwargs["plot_use"]

    def __repr__(self) -> str:
        raise NotImplementedError()

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
    
    def __set_init_pred(self) -> None:
        self.__pred_x = np.ndarray([1, self.n_states, 1])
        self.__pred_y = np.ndarray([1, self.n_outputs, 1])
        
        self.__pred_x[0] = self.__x[-1]
        self.__pred_y[0] = self._output(self.__pred_x[0], self.__u[-1])

    def __update_x_pred(self, x_pred_k: np.ndarray) -> None:
        x_pred_next = x_pred_k if type(x_pred_k) is np.ndarray else x_pred_k.full()
        self.__pred_x = np.concatenate([self.__pred_x, np.atleast_3d(x_pred_next.squeeze())], axis=0)

    def __update_y_pred(self, y_pred_k: np.ndarray) -> None:
        y_pred_next = y_pred_k if type(y_pred_k) is np.ndarray else y_pred_k.full()
        self.__pred_y = np.concatenate([self.__pred_y, np.atleast_3d(y_pred_next.squeeze())], axis=0)

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
            uk, u_pred = control_law(self.__x[-1], ref_traj[k])
            x_next = self._f(x0=self.__x[-1], p=uk)
            yk = self._output(x_next, uk)

            # Time update
            self.__update_x(x_next)
            self.__update_u(uk)
            self.__update_y(yk)

            self.__set_init_pred()

            # Make prediction of full horizon if using predictive controller
            self.__prediction_openloop(u_pred)

            if k % 10 == 0:
                # plot open-loop prediction
                self.__plot_prediction(k)

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
        plot_range = np.linspace(start, end, self.__pred_y.shape[0], endpoint=False)

        for i in range(self.n_outputs):
            axis.plot(plot_range, self.__pred_y[:, i, :], **pltargs)

    def plot_trajectory(self, **pltargs):
        pltargs.setdefault('linewidth', 1.2)
        axis = self.__plot_axis

        y = self.get_y()

        plot_range = np.linspace(
            0, y.shape[0]*self.Ts, y.shape[0], endpoint=False)

        for i in range(self.n_outputs):
            axis.plot(plot_range, y[:, i, :],
                      label=r"$y_{}$".format(i), **pltargs)

        axis.set_ylim(np.min(self.lb_output), np.max(self.ub_output))
        axis.legend()

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

    def __repr__(self) -> str:
        info = "Nonlinear system"
        return info

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

        self.observable = None
        self.controllable = None

        self._set_noise(**kwargs)
        self._set_initial_states(x0=x0)

    def __repr__(self) -> str:
        info = "Linear system"
        return info

    def _f(self, x0, p) -> np.ndarray:
        x_next = self.A@x0 + self.B@p
        return x_next

    def _output(self, x, u) -> np.ndarray:
        y = self.C @ x + self.D @ u + self._measurement_noise()
        return y

    def ctrl(self) -> None:
        A = self.A
        B = self.B
        n = A.shape[0]

        ctrl_mat = np.hstack(
            [B] + [np.linalg.matrix_power(A, i) @ B for i in range(1, n)])

    def obsv(self) -> None:
        A = self.A
        C = self.C
        n = A.shape[0]

        obsv_mat = C
        for i in range(1, n):
            obsv_mat = np.vstack([obsv_mat, C @ np.linalg.matrix_power(A, i)])
            rk = np.linalg.matrix_rank(obsv_mat)

        if np.linalg.matrix_rank(obsv_mat) == n:
            self.observable = True
        else:
            self.observable = False
