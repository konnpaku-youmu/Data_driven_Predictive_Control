import numpy as np
from typing import Tuple, Callable, overload
import casadi as cs
from casadi.casadi import sin, cos, tan
import matplotlib.pyplot as plt
from helper import forward_euler, zoh, runge_kutta, SetpointGenerator

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
    
    def set_initial_states(self, x0):
        self.x = np.ndarray([1, self.n_states, 1])
        self.y = np.ndarray([1, self.n_outputs, 1])
        self.u = np.zeros([1, self.n_inputs, 1])

        self.u[0] = np.zeros([self.n_inputs, 1])
        self.x[0] = x0
        self.y[0] = self.output(self.x[0], self.u[0])

    def update_u(self, uk: np.ndarray) -> None:
        u_next = uk if type(uk) is np.ndarray else uk.full()
        self.u = np.concatenate(
            [self.u, np.atleast_3d(u_next.squeeze())], axis=0)

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
            self.set_initial_states(x0)

        for k in range(1, n_steps):
            # if np.abs(self.y[-1, 0]) >= 5 or np.abs(self.y[-1, 1]) >= 2:
            #     break
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
            plt.plot(plot_range, self.y[:, i, :],
                     label=r"$y_{}$".format(i), **pltargs)

        plt.legend()

    def plot_control_input(self, **pltargs):
        pltargs.setdefault('linewidth', 0.7)
        pltargs.setdefault('linestyle', '--')

        plot_range = np.linspace(
            0, self.y.shape[0]*self.Ts, self.y.shape[0], endpoint=False)

        for i in range(self.n_inputs):
            plt.plot(plot_range, self.u[:, i, :],
                     label=r"$u_{}$".format(i), **pltargs)

        plt.legend()


class NonlinearSystem(System):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.dynamics = None

    def build_system_model(self, states: cs.MX, inputs: cs.MX,
                           outputs: cs.MX, C: np.array = None, **kwargs) -> None:
        # set default values for arguments
        kwargs.setdefault("noisy", False)

        try:
            self.Ts = kwargs["Ts"]
        except KeyError:
            print("Continuous dynamics")

        self.n_states = states.shape[0]
        self.n_inputs = inputs.shape[0]
        self.n_outputs = outputs.shape[0]
        self.C = C if C is not None else np.eye(self.n_states)  # output matrix
        self.sym_x = states
        self.sym_u = inputs
        self.sym_y = outputs

        self.noisy = kwargs["noisy"]
        if self.noisy:
            kwargs.setdefault("s_y", 1e-4*np.diag(np.ones(self.n_outputs)))
            self.σ_y = kwargs["s_y"]
        else:
            self.σ_y = np.diag(np.zeros(self.n_outputs))

        self.dynamics = self.define_dynamics()
        self.F = self.discrete_dynamics()

    def discrete_dynamics(self) -> cs.Function:
        DAE = {"x": self.sym_x,
               "p": self.sym_u,
               "ode": self.dynamics}
        opts = {"tf": self.Ts}
        return cs.integrator('F', 'cvodes', DAE, opts)

    def measurement_noise(self) -> np.ndarray:
        return np.random.multivariate_normal(np.zeros(self.n_outputs), self.σ_y, size=[1]).T

    def f(self, x0, p) -> np.ndarray:
        x_next = self.F(x0=x0, p=p)
        x_next = x_next['xf']
        return x_next.full().squeeze()

    def output(self, x, u) -> np.ndarray:
        return self.C @ x + self.measurement_noise()

    def plot_phasespace(self, **pltargs) -> None:
        plt.plot(self.x[:, 0, :], self.x[:, 2, :], linewidth=0.1)
        plt.axis("equal")

    def define_dynamics(self) -> cs.MX:
        ...


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
        kwargs.setdefault("mismatch", False)
        if kwargs["mismatch"]:
            self.A *= 1.02

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
        x0_dot = (3-self.sym_x[1]**2)*self.sym_x[0] - \
            self.sym_x[1] + self.sym_u
        x1_dot = self.sym_x[0]

        return cs.vertcat(x0_dot, x1_dot)


class LorenzAttractor(NonlinearSystem):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        x = cs.MX.sym('x', 3)
        u = cs.MX.sym('u', 3)
        y = cs.MX.sym('y', 3)

        self.build_system_model(x, u, y, **kwargs)

    def define_dynamics(self) -> cs.MX:
        # system equation
        x0_dot = 10 * (self.sym_x[1] - self.sym_x[0]) + self.sym_u[0]
        x1_dot = self.sym_x[0] * (28 - self.sym_x[2]) - \
            self.sym_x[1] + self.sym_u[1]
        x2_dot = self.sym_x[0] * self.sym_x[1] - \
            2.2 * self.sym_x[2] + self.sym_u[2]

        return cs.vertcat(x0_dot, x1_dot, x2_dot)


class IPNonlinear(NonlinearSystem):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        x = cs.MX.sym("x", 4)
        u = cs.MX.sym("u", 1)
        y = cs.MX.sym("y", 2)

        C = np.array([[1, 0, 0, 0],
                      [0, 1, 0, 0]])

        self.build_system_model(x, u, y, C, **kwargs)

    def define_dynamics(self) -> cs.MX:

        Rm = 2.6
        Km = 0.00767
        Kb = 0.00767
        Kg = 3.7
        M = 0.455
        l = 0.305
        m = 0.210
        r = 0.635e-2
        g = 9.81

        kv = (Km*Kg)/(Rm*r)
        kxd = (Km*Kb*(Kg**2))/(Rm*r)

        x0_dot = self.sym_x[2]
        x1_dot = self.sym_x[3]
        x2_dot = kv * self.sym_u - kxd * \
            self.sym_x[2] + m*l*(self.sym_x[3]**2) * \
            sin(self.sym_x[1]) - m*g*sin(self.sym_x[1])
        x2_dot /= M + m*(sin(self.sym_x[1])**2)
        x3_dot = g*sin(self.sym_x[1]) - x2_dot * cos(self.sym_x[1])
        x3_dot /= l

        return cs.vertcat(x0_dot, x1_dot, x2_dot, x3_dot)


class Quadcopter(NonlinearSystem):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        x = cs.MX.sym('x', 12)
        u = cs.MX.sym('u', 4)
        y = cs.MX.sym('y', 6)

        C = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]])

        self.build_system_model(x, u, y, C, **kwargs)

    def define_dynamics(self) -> cs.MX:

        m = 0.5
        l = 0.25
        k = 3e-6
        b = 1e-7
        g = 9.81
        Kd = 0.25
        Ixx = 5e-3
        Iyy = 5e-3
        Izz = 1e-2
        Cm = 1e4

        f_eq = m*g/(4*k*Cm)

        # Vitesse -- direction x,y,z
        vx, vy, vz = self.sym_x[3], self.sym_x[4], self.sym_x[5]
        # Pose
        φ, θ, ψ = self.sym_x[6], self.sym_x[7], self.sym_x[8]
        # Vitesse angulaire
        ωx, ωy, ωz = self.sym_x[9], self.sym_x[10], self.sym_x[11]

        # Entrée du système: Voltage^2
        u1, u2, u3, u4 = self.sym_u[0] + f_eq, self.sym_u[1] + f_eq, self.sym_u[2] + f_eq, self.sym_u[3] + f_eq

        # Définir la dynamique
        ax = (-Kd/m)*vx + (k*Cm/m)*(sin(ψ)*sin(φ) +
                                    cos(ψ)*cos(φ)*sin(θ)) * (u1+u2+u3+u4)
        ay = (-Kd/m)*vy + (k*Cm/m)*(cos(φ)*sin(ψ)*sin(θ) - 
                                    cos(ψ)*sin(φ)) * (u1+u2+u3+u4)
        az = (-Kd/m)*vz - g + (k*Cm/m)*cos(θ)*cos(φ) * (u1+u2+u3+u4)

        Φ = ωx + ωy*(sin(φ)*tan(θ)) + ωz*(cos(φ)*tan(θ))
        Θ = ωy * cos(φ) - ωz * sin(φ)
        Ψ = (sin(φ) / cos(θ)) * ωy + (cos(φ) / cos(θ)) * ωz

        Ωx = (l*k*Cm/Ixx) * (u1 - u3) - ((Iyy-Izz)/Ixx)*ωy*ωz
        Ωy = (l*k*Cm/Iyy) * (u2 - u4) - ((Izz-Ixx)/Iyy)*ωx*ωz
        Ωz = (b*Cm/Izz) * (u1 - u2 + u3 - u4) - ((Ixx-Iyy)/Izz)*ωx*ωy

        return cs.vertcat(vx, vy, vz, ax, ay, az, Φ, Θ, Ψ, Ωx, Ωy, Ωz)


if __name__ == "__main__":
    model = Quadcopter(Ts=0.05)

    x0 = np.zeros([12, 1])
    model.simulate(x0, 200)

    model.plot_trajectory()
    plt.show()
