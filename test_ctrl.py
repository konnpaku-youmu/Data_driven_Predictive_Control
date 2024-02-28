import matplotlib.pyplot as plt
from helper import generate_road_profile
from Controller import LQRController, MPC, OpenLoop, DeePC
import numpy as np

from multiprocessing import Pool, Lock, Manager, current_process
from functools import partial

from SysBase import *
from SysModels import ActiveSuspension, InvertedPendulum
from StateEstimator import KF


plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 12
})


def init_pool():
    np.random.seed(current_process().pid)


def main():
    dist, v, Ts = 40, 10, 0.02
    t_sim = dist / v
    n_steps = int(t_sim/Ts)

    x = np.array([[0], [0], [0], [0]])

    fig1, [ax1, ax2] = plt.subplots(1, 2)
    ax1.set_title("Acceleration")
    ax2.set_title("Control Input")

    suspension = ActiveSuspension(x0=x, Ts=Ts)

    σ_w, σ_v, σ_p = 0.025, 0.04, 0.025
    kalman = KF(suspension, x, σ_w=σ_w, σ_v=σ_v, σ_p=σ_p)

    Q_lqr = np.array([[2e3,   0, 0, 0],
                      [0,   3e4, 0, 0],
                      [0,   0, 5e2, 0],
                      [0,   0, 0, 5e2]])

    Q = np.array([[2e2,       0],
                  [0,     1.5e3]])

    R = np.array([[0.01]])

    T_ini = 4
    λ_s, λ_g = 6e2, 8.2e2
    horizon = 5

    profile, d_profile = generate_road_profile(dist, n_steps, Ts)
    # d_profile = None

    lqr = LQRController(suspension, Q=Q_lqr, R=R)
    mpc = MPC(suspension, horizon=horizon, Q=Q, R=R)

    suspension.simulate(n_steps,
                        control_law=None,
                        observer=None,
                        reference=None,
                        disturbance=d_profile)

    suspension.plot_trajectory(axis=ax1, states=[1], color="gray")

    # suspension.rst(x)
    # suspension.simulate(n_steps,
    #                     control_law=lqr,
    #                     observer=kalman,
    #                     reference=None,
    #                     disturbance=d_profile)
    # suspension.plot_trajectory(axis=ax1, states=[1])
    # suspension.plot_control_input(axis=ax2)

    suspension.rst(x)
    suspension.simulate(n_steps,
                        control_law=mpc,
                        observer=None,
                        reference=None,
                        disturbance=d_profile)
    suspension.plot_trajectory(axis=ax1, states=[1])
    suspension.plot_control_input(axis=ax2)

    suspension.rst(x)
    excitation = OpenLoop.rnd_input(suspension, n_steps)
    dpc = DeePC(suspension, T_ini=T_ini, horizon=horizon,
                init_law=excitation, λ_s=λ_s, λ_g=λ_g, Q=Q, R=R)
    suspension.rst(x)

    suspension.simulate(n_steps,
                        control_law=dpc,
                        reference=np.zeros((n_steps, suspension.p, 1)),
                        disturbance=d_profile)
    suspension.plot_trajectory(axis=ax1, states=[1], trim_exci=True)
    suspension.plot_control_input(axis=ax2, trim_exci=True)

    plt.show()


def sim_parellel(n_steps, x, Ts, d_profile, λ_s, λ_g, params):

    T_ini, init_l = params[0], params[1]

    suspension = ActiveSuspension(x0=x, Ts=Ts)

    excitation = OpenLoop.rnd_input(suspension, n_steps)
    dpc = DeePC(suspension, T_ini=T_ini, horizon=5,
                init_law=excitation, init_len=init_l,
                λ_s=λ_s, λ_g=λ_g)

    suspension.simulate(n_steps,
                        control_law=dpc,
                        reference=None,
                        disturbance=d_profile)

    loss = dpc.get_total_loss()

    return params, loss


if __name__ == "__main__":
    main()
    # map = np.load("loss_map_sg.npy")
    # map = np.ma.array(map, mask=np.isnan(map))
    # map = np.log2(map)
    # plt.matshow((map - np.mean(map))/(np.max(map) - np.min(map)))
    # plt.show()
