import matplotlib.pyplot as plt
from Controller import LQRController, MPC
import numpy as np

from SysModels import ActiveSuspension


plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 12
})


def main():
    Ts = 0.01
    n_steps = 200

    x = np.array([[0], [0], [0], [0]])

    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()

    suspension = ActiveSuspension(x0=x, Ts=Ts)

    Q = np.array([[80,   0,  0,   0],
                  [0,  2e3,  0,   0],
                  [0,    0, 50,   0],
                  [0,    0,  0, 200]])
    R = np.array([[1]])

    lqr = LQRController(suspension, Q=Q, R=R)
    mpc = MPC(suspension, horizon=20, Q=Q, R=R)

    road_profile = np.zeros((n_steps, 1))
    road_profile[50] = 5
    road_profile[51] = 5

    suspension.simulate(n_steps, control_law=mpc,
                        reference=None, disturbance=road_profile)

    suspension.plot_trajectory(axis=ax1)
    suspension.plot_control_input(axis=ax2)
    plt.show()


if __name__ == "__main__":
    main()
