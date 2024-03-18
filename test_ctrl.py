import matplotlib.pyplot as plt
from helper import generate_road_profile
from Controller import LQRController, MPC, OpenLoop, DeePC
import numpy as np

from multiprocessing import Pool, Lock, Manager, current_process
from functools import partial

from SysBase import *
from SysModels import ActiveSuspension, SimpleBicycle
from StateEstimator import KF

from rcracers.utils.geometry import plot_polytope

plt.rcParams.update({
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{nicefrac, xfrac} \boldmath",
    "font.family": "serif",
    "font.size": 12,
    "font.weight": "bold",
    "axes.linewidth": 1
})

plt.rcParams['figure.dpi'] = 100


def init_pool():
    np.random.seed(current_process().pid)


def test_passive():
    dist, v, Ts = 50, 5, 0.05
    t_sim = dist / v
    n_steps = int(t_sim/Ts)

    x = np.array([[0], [0], [0], [0]])

    suspension = ActiveSuspension(x0=x, Ts=Ts)

    plt.figure(figsize=(14, 6))

    ax1 = plt.subplot(1, 2, 1)
    ax2 = plt.subplot(2, 2, 2)
    ax3 = plt.subplot(2, 2, 4)

    profile, d_profile = generate_road_profile(dist, n_steps, Ts)

    ax1.set_title(r"\textbf{Road profile}")
    ax1.set_ylabel(r"{Height ($m$)}")
    ax1.set_ylim(-0.05, 0.2)
    ax1.set_xlabel(r"{Time(s)}")
    ax1.plot(np.linspace(0, n_steps*Ts, n_steps+1, endpoint=False),
             profile, color="red", label=r"$w$")
    ax1.legend()

    suspension.simulate(n_steps,
                        control_law=None,
                        observer=None,
                        reference=None,
                        disturbance=d_profile)
    ax2.set_title(r"\textbf{Suspension displacement}")
    ax2.set_ylabel(r"{Displacement ($m$)}")
    suspension.plot_trajectory(axis=ax2, states=[0])

    ax3.sharex(ax2)
    ax3.set_title(r"\textbf{Body Acceleration}")
    ax3.set_ylabel(r"{Acceleration ($\sfrac{m}{s^2}$)}")
    suspension.plot_trajectory(axis=ax3, states=[1])

    plt.tight_layout()
    plt.show()


def test_lqr():
    dist, v, Ts = 50, 5, 0.05
    t_sim = dist / v
    n_steps = int(t_sim/Ts)

    x = np.array([[0], [0], [0], [0]])

    plt.figure(figsize=(14, 6))

    ax1 = plt.subplot(2, 2, 1)
    ax2 = plt.subplot(2, 2, 3)
    ax3 = plt.subplot(1, 2, 2)

    ax2.sharex(ax1)

    suspension = ActiveSuspension(x0=x, Ts=Ts)

    σ_w, σ_v, σ_p = 0.01, 0.04, 0.025
    kalman = KF(suspension, x, σ_w=σ_w, σ_v=σ_v, σ_p=σ_p)

    Q_lqr = np.diag([2e2, 1e4, 0, 0])
    R = np.diag([0.01])

    profile, d_profile = generate_road_profile(dist, n_steps, Ts)

    lqr = LQRController(suspension, Q=Q_lqr, R=R)

    suspension.simulate(n_steps,
                        control_law=None,
                        observer=None,
                        reference=None,
                        disturbance=d_profile)
    ax1.set_title(r"\textbf{Suspension displacement}")
    ax1.set_ylabel(r"{Displacement ($m$)}")
    suspension.plot_trajectory(axis=ax1, states=[0],
                               color="gray",
                               label_prefix=r"Passive")

    ax2.set_title(r"\textbf{Body Acceleration}")
    ax2.set_ylabel(r"{Acceleration ($\sfrac{m}{s^2}$)}")
    suspension.plot_trajectory(axis=ax2, states=[1],
                               color="gray",
                               label_prefix=r"Passive")

    Q_11 = [1e4, 2e4, 4e4, 8e4]

    for q in Q_11:
        suspension.rst(x)

        Q_lqr[1, 1] = q
        lqr.set_weights(Q_lqr, R)

        suspension.simulate(n_steps,
                            control_law=lqr,
                            observer=None,
                            reference=None,
                            disturbance=d_profile)
        ax1.set_title(r"\textbf{Suspension displacement}")
        ax1.set_ylabel(r"{Displacement ($m$)}")
        suspension.plot_trajectory(axis=ax1, states=[0],
                                   label_prefix=r"$Q = {}$".format(q))

        ax2.set_title(r"\textbf{Body Acceleration}")
        ax2.set_ylabel(r"{Acceleration ($\sfrac{m}{s^2}$)}")
        suspension.plot_trajectory(axis=ax2, states=[1],
                                   label_prefix=r"$Q = {}$".format(q))

        ax3.set_title(r"\textbf{Actuator force}")
        ax3.set_ylabel(r"{Force ($N$)}")
        suspension.plot_control_input(axis=ax3)

    plt.tight_layout()
    plt.show()


def test_all():
    dist, v, Ts = 50, 5, 0.05
    t_sim = dist / v
    n_steps = int(t_sim/Ts)

    x = np.array([[0], [0], [0], [0]])

    fig1 = plt.figure(figsize=(14, 6))
    ax1 = fig1.add_subplot(2, 2, 1)
    ax2 = fig1.add_subplot(2, 2, 3)
    ax3 = fig1.add_subplot(1, 2, 2)
    fig1.tight_layout()

    ax2.sharex(ax1)
    ax3.sharex(ax1)

    fig2 = plt.figure(figsize=(9, 4))
    ax_loss = fig2.add_subplot(1, 1, 1)
    fig2.tight_layout()

    suspension = ActiveSuspension(x0=x, Ts=Ts)

    σ_w, σ_v, σ_p = 0.025, 0.04, 0.025
    kalman = KF(suspension, x, σ_w=σ_w, σ_v=σ_v, σ_p=σ_p)

    Q_lqr = np.diag([2e2, 8e4, 10, 10])
    Q_pc = np.diag([2e2, 4e3])
    R = np.diag([0.005])

    profile, d_profile = generate_road_profile(dist, n_steps, Ts)

    ax1.set_title(r"\textbf{Suspension displacement}")
    ax1.set_ylabel(r"{Displacement ($m$)}")
    ax2.set_title(r"\textbf{Body Acceleration}")
    ax2.set_ylabel(r"{Acceleration ($\sfrac{m}{s^2}$)}")
    ax3.set_title(r"\textbf{Actuator force}")
    ax3.set_ylabel(r"{Force ($N$)}")

    ### LQR ###
    # lqr = LQRController(suspension, Q=Q_lqr, R=R)

    # suspension.rst(x)
    # suspension.simulate(n_steps,
    #                     control_law=lqr,
    #                     observer=None,
    #                     reference=None,
    #                     disturbance=d_profile)

    # suspension.plot_trajectory(axis=ax1, states=[0],
    #                             label_prefix=r"LQR")
    # suspension.plot_trajectory(axis=ax2, states=[1],
    #                             label_prefix=r"LQR")
    # suspension.plot_control_input(axis=ax3)

    ### MPC ###
    horizon = 5
    mpc = MPC(suspension, horizon=horizon, Q=Q_pc, R=R, enforce_term=False)

    suspension.rst(x)
    suspension.simulate(n_steps,
                        control_law=mpc,
                        observer=kalman,
                        reference=None,
                        disturbance=d_profile)

    suspension.plot_trajectory(axis=ax1, states=[0],
                               label_prefix=r"MPC$_{{{}}}$".format(horizon))
    suspension.plot_trajectory(axis=ax2, states=[1],
                               label_prefix=r"MPC$_{{{}}}$".format(horizon))
    suspension.plot_control_input(axis=ax3)

    mpc.plot_loss(ax_loss)

    ### DeePC ###
    T_ini = 5
    λ_s, λ_g = 5e3, 2e2

    suspension.rst(x)
    excitation = OpenLoop.rnd_input(suspension, n_steps)
    dpc = DeePC(suspension, T_ini=T_ini, horizon=horizon,
                init_law=excitation, λ_s=λ_s, λ_g=λ_g, Q=Q_pc, R=R)
    suspension.rst(x)

    suspension.simulate(n_steps,
                        control_law=dpc,
                        reference=np.zeros((n_steps, suspension.p, 1)),
                        disturbance=d_profile)
    suspension.plot_trajectory(axis=ax1, states=[0], trim_exci=True,
                               label_prefix=r"DeePC$_{{{0}}}$, $T_{{ini}}$ = {{{1}}}".format(horizon, T_ini))
    suspension.plot_trajectory(axis=ax2, states=[1], trim_exci=True,
                               label_prefix=r"DeePC$_{{{0}}}$, $T_{{ini}}$ = {{{1}}}".format(horizon, T_ini))
    suspension.plot_control_input(axis=ax3, trim_exci=True)

    dpc.plot_loss(ax_loss)
    dpc.plot_data_matt_svd()

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


def test_simple_bicycle():
    Ts = 0.05
    n_steps = 100
    x = np.array([[0], [0], [0], [0]])

    fig1 = plt.figure(figsize=(14, 6))
    ax1 = fig1.add_subplot(1, 2, 1)
    ax3 = fig1.add_subplot(1, 2, 2)
    fig1.tight_layout()

    vehicle = SimpleBicycle(x0=x, Ts=Ts)
    u = np.vstack([0.5*np.ones(n_steps),
                  [0.1 * np.sin(0.5*np.pi*np.linspace(0, Ts*n_steps, n_steps))]]).T

    test_policy = OpenLoop.given_input_seq(vehicle, u)

    vehicle.simulate(n_steps=n_steps,
                     control_law=test_policy)
    vehicle.plot_phasespace(axis=ax1, states=[0, 1])
    vehicle.plot_control_input(axis=ax3)

    plt.show()


def simple_bicycle_mpc():
    Ts, n_steps = 0.05, 100

    x = np.array([[0.6], [-0.3], [0], [0]])

    fig1 = plt.figure(figsize=(14, 6))
    ax1 = fig1.add_subplot(1, 2, 1)
    ax3 = fig1.add_subplot(1, 2, 2)
    fig1.tight_layout()

    vehicle = SimpleBicycle(x0=x, Ts=Ts)

    horizon = 20
    Q = np.diag([1.5, 10, 0.06, 0.01])
    R = np.diag([1, 0.1])

    mpc = MPC(vehicle,
              horizon=horizon,
              Q=Q, R=R)

    vehicle.simulate(n_steps=n_steps,
                     control_law=mpc)
    vehicle.plot_phasespace(axis=ax1, states=[0, 1])
    vehicle.plot_control_input(axis=ax3)

    vehicle.rst(x0=x)
    T_ini = 25
    λ_s, λ_g = 2e3, 1e3
    excitation = OpenLoop.rnd_input(vehicle, n_steps)

    dpc = DeePC(vehicle, T_ini=T_ini,
                Q=Q, R=R,
                λ_s=λ_s, λ_g=λ_g,
                horizon=horizon,
                init_law=excitation)

    vehicle.rst(x0=x)
    vehicle.simulate(n_steps,
                     control_law=dpc,
                     reference=np.zeros((n_steps, vehicle.p, 1)))

    vehicle.plot_phasespace(axis=ax1, states=[0, 1])
    vehicle.plot_control_input(axis=ax3)

    plt.show()

    return


if __name__ == "__main__":
    # test_simple_bicycle()
    simple_bicycle_mpc()
