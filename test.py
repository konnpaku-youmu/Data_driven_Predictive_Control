import numpy as np

from System import SimpleHarmonic, InvertedPendulum, VanderpolOscillator, LorenzAttractor, IPNonlinear, Quadcopter
from Controller import LQRController, OpenLoop, SetpointGenerator, DeePC

import matplotlib.pyplot as plt


def main():
    model = InvertedPendulum(Ts=0.05, noisy=True)
    x0 = np.array([[0.5], [0.], [0.], [0.]])
    horizon = 20
    n_steps = 500

    Q = np.array([[1.5, 0, 0, 0],
                  [0, 10, 0, 0],
                  [0, 0, 0, 0],
                  [0, 0, 0, 0]])

    lqr = LQRController(model, Q=Q, R=0.005)

    Q_dpc = np.array([[185, 0],
                      [0,  120]])
    R_dpc = 0.01
    exct_bounds = np.array([[[-0.5], [0.5]]])

    cov = np.array([1e-4, 1e-4])
    nl_model = IPNonlinear(Ts=0.05, noisy=True, s_y=np.diag(cov))
    deepc = DeePC(nl_model, T_ini=4, N=horizon, data_mat="page",
                  init_law=lqr, excitation_bounds=exct_bounds,
                  Q=Q_dpc, R=R_dpc)

    lbx = np.array([[-1], [-0.3]])
    ubx = np.array([[1], [0.3]])
    lbu = np.array([[-5]])
    ubu = np.array([[5]])
    deepc.build_controller(lbx=lbx, ubx=ubx, lbu=lbu, ubu=ubu)
    sp_gen = SetpointGenerator(nl_model.n_outputs, n_steps,
                               nl_model.Ts, 0, "rand", exct_bounds, switching_prob=0.01)
    nl_model.simulate(x0, n_steps, control_law=deepc, tracking_target=sp_gen())

    plt.figure()
    nl_model.plot_trajectory()
    plt.ylim(-1, 1)
    # model.plot_control_input()
    deepc.plot_reference()

    plt.show()


def run_vanderpol():
    model = VanderpolOscillator(Ts=0.05, noisy=True)
    x0 = np.array([[0.1], [0.5]])
    horizon = 15
    n_steps = 1000

    ctrl = OpenLoop(model)
    ctrl.generate_rnd_input_seq(200, np.array([[-10]]), np.array([[10]]))

    Q_dpc = np.array([[15, 0],
                      [0,  21]])
    R_dpc = 0.01
    exct_bounds = np.array([[[-5], [5]]])
    deepc = DeePC(model, T_ini=8, N=horizon, data_mat="hankel",
                  init_law=ctrl, excitation_bounds=exct_bounds, Q=Q_dpc, R=R_dpc)

    lbx = np.array([[-7], [-7]])
    ubx = np.array([[7], [7]])
    lbu = np.array([[-50]])
    ubu = np.array([[50]])
    deepc.build_controller(lbx=lbx, ubx=ubx, lbu=lbu, ubu=ubu)

    exct_bounds = np.array([[[-5], [5]]])
    sp_gen = SetpointGenerator(model.n_outputs, n_steps, model.Ts, [0], [
                               "rand"], exct_bounds, switching_prob=0.03)
    model.simulate(x0, n_steps, control_law=deepc, tracking_target=sp_gen())

    model.plot_trajectory()
    model.plot_control_input()
    deepc.plot_reference()
    plt.show()


def run_lorenz():
    model = LorenzAttractor(Ts=0.05, noisy=True)
    x0 = np.array([[4], [5], [2]])
    horizon = 15
    n_steps = 500

    ctrl = OpenLoop(model)
    ctrl.generate_rnd_input_seq(250, np.array(
        [[-100], [-100], [-100]]), np.array([[100], [100], [100]]))

    Q_dpc = np.array([[23, 2,   7],
                      [2,  25,  0],
                      [7,   0,  1]])
    R_dpc = np.array([[0.1, 0, 0],
                      [0, 0.1, 0],
                      [0, 0, 0.1]])

    exct_bounds = np.array([[[-50], [50]]])
    deepc = DeePC(model, T_ini=8, N=horizon, data_mat="page",
                  init_law=ctrl, excitation_bounds=exct_bounds, Q=Q_dpc, R=R_dpc)

    lbx = np.array([[-100], [-100], [-100]])
    ubx = np.array([[100], [100], [100]])
    lbu = np.array([[-500], [-500], [-500]])
    ubu = np.array([[500], [500], [500]])
    deepc.build_controller(lbx=lbx, ubx=ubx, lbu=lbu, ubu=ubu)

    exct_bounds = np.array([[[-20], [20]], [[-20], [20]], [[-20], [20]]])
    sp_gen = SetpointGenerator(model.n_outputs, n_steps, model.Ts, [0, 1, 2], [
                               "rand", "rand", "rand"], exct_bounds, switching_prob=0.02)
    model.simulate(x0, n_steps, control_law=deepc, tracking_target=sp_gen())

    model.plot_trajectory()
    # model.plot_control_input()
    deepc.plot_reference()
    plt.show()


def run_Quadcopter():
    m = 0.5
    k = 3e-6
    g = 9.81
    Cm = 1e4

    f = m*g/(4*k*Cm)

    cov = np.array([1e-4, 1e-4, 1e-4, 1e-6, 1e-6, 1e-6])
    model = Quadcopter(Ts=0.05, noisy=True, s_y=np.diag(cov))

    openloop = OpenLoop(model)

    len = 1000
    u = np.ones([len, 4, 1]) * f
    openloop.generate_rnd_input_seq(len, np.array([[-0.01], [-0.01], [-0.01], [-0.01]]), np.array([[0.01], [0.01], [0.01], [0.01]]), switch_prob=0.1)
    openloop.u = u

    x0 = np.zeros([12, 1])
    x0[2] = 1
    model.simulate(x0, len, control_law=openloop)

    model.plot_trajectory()
    # model.plot_control_input()
    plt.ylim(-2, 2)
    plt.show()

if __name__ == "__main__":
    run_Quadcopter()
