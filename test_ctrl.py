import matplotlib.pyplot as plt
from Controller import LQRController, SMStruct
import numpy as np

from SysModels import ActiveSuspension


plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 12
})


def main():
    Ts = 0.05
    n_steps = 100

    x = np.array([[0.015625], [0], [-1.84e-3], [0]])

    fig, ax1 = plt.subplots()

    suspension = ActiveSuspension(x0=x, plot_use=ax1, Ts=Ts)

    Q = np.array([[20,  0, 0, 0],
                  [0,  100, 0, 0],
                  [0,  0, 1, 0],
                  [0,  0, 0, 5]])
    R = np.array([[0.01]])

    controller = LQRController(suspension, Q=Q, R=R)

    road_profile = np.zeros((n_steps, 1))
    road_profile[30] = 2
    road_profile[50] = 0

    suspension.simulate(n_steps, control_law=None, reference=None, disturbance=road_profile)

    suspension.plot_trajectory()
    plt.show()

if __name__ == "__main__":
    main()
