import matplotlib.pyplot as plt
from helper import generate_road_profile
from Controller import LQRController, MPC, OpenLoop, DeePC
import numpy as np

from SysModels import ActiveSuspension


plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 12
})


def main():
    dist, v, Ts = 40, 10, 0.02
    t_sim = dist / v
    n_steps = int(t_sim/Ts)

    x = np.array([[0], [0], [0], [0]])

    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()

    suspension = ActiveSuspension(x0=x, Ts=Ts)

    Q = np.array([[20,   0,  0,   0],
                  [0,   500,  0,   0],
                  [0,    0, 100,   0],
                  [0,    0,  0, 50]])
    R = np.array([[0.01]])

    profile, d_profile = generate_road_profile(dist, n_steps, Ts)

    lqr = LQRController(suspension, Q=Q, R=R)
    mpc = MPC(suspension, horizon=5, Q=Q, R=R)

    suspension.simulate(n_steps, control_law=None,
                        reference=None, disturbance=d_profile)
    suspension.plot_trajectory(axis=ax1, states=[1])

    # suspension.rst(x)
    # suspension.simulate(n_steps, control_law=lqr,
    #                     reference=None, disturbance=d_profile)
    # suspension.plot_trajectory(axis=ax1, states=[1])

    # suspension.rst(x)
    # suspension.simulate(n_steps, control_law=mpc,
    #                     reference=None, disturbance=d_profile)
    # suspension.plot_trajectory(axis=ax1, states=[1])
    suspension.rst(x)
    λ_s, λ_g = 15, 25
    excitation = OpenLoop.rnd_input(suspension, n_steps)
    dpc = DeePC(suspension, 4, 10, excitation, λ_s=λ_s, λ_g=λ_g)

    suspension.simulate(n_steps,
                        control_law=dpc,
                        reference=None,
                        disturbance=d_profile)

    suspension.plot_trajectory(axis=ax1, states=[1])

    ax1.plot(np.linspace(0, t_sim, n_steps), profile[:-1])
    suspension.plot_control_input(axis=ax2)

    plt.show()


if __name__ == "__main__":
    main()
