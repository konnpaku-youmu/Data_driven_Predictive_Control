import numpy as np

from System import SimpleHarmonic, InvertedPendulum
from Controller import LQRController, OpenLoop, SetpointGenerator, DeePC

import matplotlib.pyplot as plt


def main():

    model = InvertedPendulum(Ts=0.05)
    x0 = np.array([[0.1], [0.], [0.], [0.]])
    n_steps = 100

    Q = np.array([[1.5, 0, 0, 0],
                  [0, 10, 0, 0],
                  [0, 0, 0, 0],
                  [0, 0, 0, 0]])

    lqr = LQRController(model, Q=Q, R=0.005)

    exct_bounds = np.array([[[-1], [1]]])
    deepc = DeePC(model, T_ini=4, N=50, init_law=lqr,
                  excitation_bounds=exct_bounds)

    deepc.build_controller()

    # model.plot_trajectory()
    # model.plot_control_input()

    plt.show()


if __name__ == "__main__":
    main()
