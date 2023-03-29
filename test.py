import matplotlib.pyplot as plt
from Controller import LQRController, OpenLoop, SetpointGenerator, DeePC, CrappyPID
import numpy as np

from System import SimpleHarmonic, InvertedPendulum, VanderpolOscillator, LorenzAttractor, IPNonlinear, Quadcopter


def main():
    model = InvertedPendulum(Ts=0.05, noisy=False, mismatch=False)
    x0 = np.array([[0.5], [0.], [0.], [0.]])
    horizon = 20
    n_steps = 500

    Q = np.array([[1.5, 0, 0, 0],
                  [0, 10, 0, 0],
                  [0, 0, 0, 0],
                  [0, 0, 0, 0]])

    lqr = LQRController(model, Q=Q, R=0.005)

    Q_dpc = np.array([[15, 0],
                      [0, 12]])
    R_dpc = 0.005
    exct_bounds = np.array([[[-1], [1]]])

    cov = np.array([1e-4, 1e-4])
    nl_model = IPNonlinear(Ts=0.05, noisy=True, s_y=np.diag(cov))

    deepc = DeePC(nl_model, T_ini=4, N=horizon, data_mat="hankel",
                  init_law=lqr, exc_bounds=exct_bounds,
                  exc_states=[0], exc_shapes=["rand"],
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
    plt.ylim(-1.5, 1.5)
    # nl_model.plot_control_input()
    deepc.plot_reference()

    plt.show()


def run_vanderpol():
    model = VanderpolOscillator(Ts=0.05, noisy=False)
    x0 = np.array([[0.1], [0.5]])
    horizon = 15
    n_steps = 200

    ctrl = OpenLoop(model)
    ctrl.generate_rnd_input_seq(200, np.array([[-10]]), np.array([[10]]))

    Q_dpc = np.array([[15, 0],
                      [0,  21]])
    R_dpc = 0.01
    exct_bounds = np.array([[[-5], [5]]])
    deepc = DeePC(model, T_ini=8, N=horizon, data_mat="hankel",
                  exc_bounds=exct_bounds,
                  exc_states=[0], exc_shapes=["rand"],
                  init_law=ctrl, Q=Q_dpc, R=R_dpc)

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

    Q_dpc = np.array([[23,  2,  7],
                      [ 2, 25,  0],
                      [ 7,  0, 10]])
    R_dpc = np.array([[0.1, 0,   0],
                      [0,  0.1,  0],
                      [0,   0, 0.1]])

    exct_bounds = np.array([[[-50], [50]]])
    deepc = DeePC(model, T_ini=8, N=horizon, data_mat="page",
                  init_law=ctrl, exc_bounds=exct_bounds, Q=Q_dpc, R=R_dpc)

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
    horizon = 10
    n_steps = 200

    # cov = np.array([1e-4, 1e-4, 1e-4, 1e-6, 1e-6, 1e-6])
    model = Quadcopter(Ts=0.05, noisy=False)

    pid = CrappyPID(model)

    exct_bounds = np.array([[[-0.5], [0.5]],
                            [[-0.5], [0.5]],
                            [[-0.5], [0.5]],
                            [[-0.2], [0.2]]])
    x0 = np.zeros([12, 1])

    Q = np.array([[1.5, 0, 0, 0, 0, 0],
                  [0, 1.5, 0, 0, 0, 0],
                  [0, 0, 2, 0, 0, 0],
                  [0, 0, 0, 35, 0, 0],
                  [0, 0, 0, 0, 35, 0],
                  [0, 0, 0, 0, 0, 12.5]])

    deepc = DeePC(model, T_ini=12, N=horizon, data_mat="hankel",
                  init_law=pid, exc_bounds=exct_bounds,
                  exc_states=[0, 1, 2, 8], exc_shapes=["rand"]*4, Q=Q)

    lbx = np.array([[-10], [-10], [-10], [-1], [-1], [-np.inf]])
    ubx = np.array([[10], [10], [100], [1], [1], [np.inf]])
    lbu = np.array([[-50], [-50], [-50], [-50]])
    ubu = np.array([[50], [50], [50], [50]])
    deepc.build_controller(lbx=lbx, ubx=ubx, lbu=lbu, ubu=ubu)

    sp = SetpointGenerator(model.n_outputs, n_steps, model.Ts,
                           [0, 1, 2, 5], ["step"]*4, exct_bounds, switch_prob=0.0001)

    model.simulate(x0, n_steps, control_law=deepc, tracking_target=sp())

    model.plot_trajectory()
    # model.plot_control_input()
    # deepc.plot_reference()
    plt.show()


if __name__ == "__main__":
    run_vanderpol()
