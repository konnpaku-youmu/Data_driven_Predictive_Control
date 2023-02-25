import numpy as np

from System import SimpleHarmonic, InvertedPendulum
from Controller import LQRController, OpenLoop

import matplotlib.pyplot as plt


def main():

    # model = SimpleHarmonic(k=1, mass=1, Ts=0.05)
    model = InvertedPendulum(Ts = 0.05)
    x0 = np.array([[0.2], [0.3], [0.], [0.]])
    n_steps = 200

    target = np.zeros([n_steps, x0.shape[0], x0.shape[1]])
    target[:100, 0, 0] = np.ones(100)

    lqr = LQRController(model)
    openloop = OpenLoop(model)
    input_lim = np.array([[0.1]])
    openloop.generate_rnd_input_seq(n_steps, input_lim, -input_lim)

    model.simulate(x0, n_steps, control_law=lqr, tracking_target=target)
    model.plot_trajectory()
    model.plot_control_input()
    plt.show()


if __name__ == "__main__":
    main()
