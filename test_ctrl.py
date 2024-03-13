import matplotlib.pyplot as plt
from helper import generate_road_profile
from Controller import LQRController, MPC, OpenLoop, DeePC
import numpy as np

from multiprocessing import Pool, Lock, Manager, current_process
from functools import partial

from SysBase import *
from SysModels import ActiveSuspension
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
    # d_profile = None

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

    plt.figure(figsize=(14, 6))

    ax1 = plt.subplot(2, 2, 1)
    ax2 = plt.subplot(2, 2, 3)
    ax3 = plt.subplot(1, 2, 2)

    ax2.sharex(ax1)

    suspension = ActiveSuspension(x0=x, Ts=Ts)

    σ_w, σ_v, σ_p = 0.01, 0.04, 0.025
    kalman = KF(suspension, x, σ_w=σ_w, σ_v=σ_v, σ_p=σ_p)

    Q_lqr = np.diag([2e2, 8e4, 0, 0])
    R = np.diag([0.01])

    profile, d_profile = generate_road_profile(dist, n_steps, Ts)
    # d_profile = None

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

    suspension.rst(x)
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

def main():
    dist, v, Ts = 50, 5, 0.05
    t_sim = dist / v
    n_steps = int(t_sim/Ts)

    x = np.array([[0], [0], [0], [0]])

    fig1, [ax1, ax2] = plt.subplots(1, 2)
    ax1.set_title("Acceleration")
    ax2.set_title("Control Input")

    suspension = ActiveSuspension(x0=x, Ts=Ts)

    σ_w, σ_v, σ_p = 0.025, 0.04, 0.025
    kalman = KF(suspension, x, σ_w=σ_w, σ_v=σ_v, σ_p=σ_p)

    Q_lqr = np.array([[2e2,   0, 0, 0],
                      [0,   4e4, 0, 0],
                      [0,   0, 50, 0],
                      [0,   0, 0, 50]])

    Q = np.array([[2e2,       0],
                  [0,       2e3]])

    R = np.array([[0.005]])

    T_ini = 6
    λ_s, λ_g = 6e2, 5e2
    horizon = 20

    profile, d_profile = generate_road_profile(dist, n_steps, Ts)
    # d_profile = None

    lqr = LQRController(suspension, Q=Q_lqr, R=R)
    mpc = MPC(suspension, horizon=horizon, Q=Q, R=R, enforce_term=False)

    suspension.simulate(n_steps,
                        control_law=None,
                        observer=None,
                        reference=None,
                        disturbance=d_profile)
    suspension.plot_trajectory(axis=ax1, states=[1], color="gray")

    suspension.rst(x)
    suspension.simulate(n_steps,
                        control_law=lqr,
                        observer=None,
                        reference=None,
                        disturbance=d_profile)
    suspension.plot_trajectory(axis=ax1, states=[1])
    suspension.plot_control_input(axis=ax2)

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

    # print(dpc.get_total_loss())

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
    test_lqr()
