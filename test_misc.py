import matplotlib.pyplot as plt
from Controller import OpenLoop, RndSetpoint, DeePC, CrappyPID
import numpy as np

from SysModels import IPNonlinear, Quadcopter, FlexJoint
from helper import hankelize

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 12
})

def main():
    T_ini = 4
    pred_horizon = 10
    n_steps = 50

    x0 = np.array([[0], [0.2], [0], [0]])

    _, ax1 = plt.subplots(1, 1, sharex=True)

    flex_joint = FlexJoint(Ts=0.05, x0=x0, plot_use=ax1)
    ctrl_open = OpenLoop.rnd_input(model=flex_joint, length=n_steps)

    flex_joint.simulate(n_steps, ctrl_open)

    y = flex_joint.get_y()
    y0 = y[:, 0:1, :]
    print(y0.shape)
    Hy = hankelize(y0, 15)
    print(Hy.shape)
    print(np.linalg.matrix_rank(Hy))

    # flex_joint.plot_trajectory()
    # plt.show()


if __name__ == "__main__":
    main()
 