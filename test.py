import numpy as np

from System import SimpleHarmonic, InvertedPendulum
from Controller import LQRController, OpenLoop, SetpointGenerator, DeePC

import matplotlib.pyplot as plt


def main():
    model = InvertedPendulum(Ts=0.05, noisy=True)
    x0 = np.array([[0.5], [0.], [0.], [0.]])
    horizon = 20
    n_steps = 200

    Q = np.array([[1.5, 0, 0, 0],
                  [0, 10, 0, 0],
                  [0, 0, 0, 0],
                  [0, 0, 0, 0]])

    lqr = LQRController(model, Q=Q, R=0.005)

    Q_dpc = np.array([[25, 0], 
                      [0,  4]])
    R_dpc = 0.1
    exct_bounds = np.array([[[-0.5], [0.5]]])
    deepc = DeePC(model, T_ini=8, N=horizon, data_mat="hankel", 
                  init_law=lqr, excitation_bounds=exct_bounds, 
                  Q=Q_dpc, R=R_dpc)

    deepc.build_controller()
    sp_gen = SetpointGenerator(model.n_output, n_steps, model.Ts, 0, "rand", exct_bounds, switching_prob=0.01)
    model.simulate(x0, n_steps, control_law=deepc, tracking_target=sp_gen())

    model.plot_trajectory()
    # model.plot_control_input()
    deepc.plot_reference()

    plt.show()
 

if __name__ == "__main__":
    main()
