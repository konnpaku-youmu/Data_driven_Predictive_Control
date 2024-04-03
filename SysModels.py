import numpy as np
import casadi as cs

from SysBase import *

class ActiveSuspension(LinearSystem):
    def __init__(self, x0: np.ndarray, **kwargs) -> None:

        k1 = 35000
        k2 = 190000
        b1 = 1000
        b2 = 2
        m1 = 375
        m2 = 59

        A = np.array([[0.,     1,           0,             -1],
                      [-k1/m1, -b1/m1,      0,          b1/m1],
                      [0,       0,          0,              1],
                      [k1/m2, b1/m2,   -k2/m2,    -(b1+b2)/m2]])

        B = np.array([[0.],
                      [1/m1],
                      [0.],
                      [-1/m2]])

        B2 = np.array([[0.],
                       [0.],
                       [-1.],
                       [b2/m2]])

        C = np.array([[1.,     0.,           0,             0],
                      [-k1/m1, -b1/m1,      0,          b1/m1]])

        D = np.array([[0.],
                      [1/m1]])

        σ_x = np.diag([0, 0, 0, 0])
        σ_y = np.diag([1e-4, 1e-2])

        super().__init__(A, B, B2, C, D, x0,
                         noisy=False, σ_x=σ_x, σ_y=σ_y,
                         **kwargs)
        
        self.output_constraint.lb[0] = -0.127
        self.output_constraint.ub[0] = 0.127
        self.output_constraint.lb[1] = -8
        self.output_constraint.ub[1] = 8

        self.input_constraint.lb[0] = -1000
        self.input_constraint.ub[0] = 1000

        self.input_names = [r"$F$"]
        self.output_names = [r"$\Delta x_{s}$", r"$\ddot{x}_1$"]


class SimpleBicycle(NonlinearSystem):
    length: float = 0.17  # length of the car (meters)
    axis_front: float = 0.047  # distance cog and front axis (meters)
    axis_rear: float = 0.05  # distance cog and rear axis (meters)
    front: float = 0.08  # distance cog and front (meters)
    rear: float = 0.08  # distance cog and rear (meters)
    width: float = 0.08  # width of the car (meters)
    height: float = 0.055  # height of the car (meters)
    mass: float = 0.1735  # mass of the car (kg)
    inertia: float = 18.3e-5  # moment of inertia around vertical (kg*m^2)

    """Pacejka 'Magic Formula' parameters.
        Used for magic formula: `peak * sin(shape * arctan(stiffness * alpha))`
        as in Pacejka (2005) 'Tyre and Vehicle Dynamics', p. 161, Eq. (4.6)
        """
    # front
    bf: float = 3.1355  # front stiffness factor
    cf: float = 2.1767  # front shape factor
    df: float = 0.4399  # front peak factor

    # rear
    br: float = 2.8919  # rear stiffness factor
    cr: float = 2.4431  # rear shape factor
    dr: float = 0.6236  # rear peak factor

    # kinematic approximation
    friction: float = 1  # friction parameter
    acceleration: float = 2  # maximum acceleration

    # motor parameters
    cm1: float = 0.3697
    cm2: float = 0.001295
    cr1: float = 0.1629
    cr2: float = 0.02133

    def __init__(self, x0: np.ndarray, **kwargs) -> None:

        x = cs.SX.sym("x", 6)
        u = cs.SX.sym("u", 2)
        y = cs.SX.sym("y", 4)
        w = cs.SX.sym("w", 4)

        C = np.eye(4, 6)

        super().__init__(x, u, y, x0, C, w=w, **kwargs)

        self.input_names = [r"$\Delta T$", r"$\Delta \delta$"]
        self.output_names = [r"$x$", r"$y$", r"$\psi$", r"$v$"]

        self.state_constraint.lb[3] = 0
        self.state_constraint.ub[3] = 2  # maximum drive
        self.state_constraint.lb[-2] = -1
        self.state_constraint.ub[-2] = 1  # maximum drive
        self.state_constraint.lb[-1] = -0.384
        self.state_constraint.ub[-1] = 0.384  # max steering angle (radians)

    def _dynamics_num(self, x, u, w) -> cs.SX:
        '''
        x: [x, y, ψ, v, T, δ]
        '''
        lf, lr = self.axis_front, self.axis_rear
        a, μ = self.acceleration, self.friction

        β = cs.arctan2(lf*cs.tan(x[5]), lf + lr)

        x_dot = x[3] * cs.cos(x[2] + β)
        y_dot = x[3] * cs.sin(x[2] + β)
        ψ_dot = x[3] * cs.sin(β) / lr
        v_dot = a * x[4] - μ * x[3]
        T_dot = u[0]
        δ_dot = u[1]

        return cs.vertcat(x_dot, y_dot, ψ_dot, v_dot, T_dot, δ_dot)
