import matplotlib.pyplot as plt
from helper import generate_road_profile
from Controller import LQRController, MPC, OpenLoop, DeePC
import numpy as np

from multiprocessing import Pool, freeze_support
from functools import partial

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

    # fig1, ax1 = plt.subplots()
    # fig2, ax2 = plt.subplots()

    # suspension = ActiveSuspension(x0=x, Ts=Ts)

    Q = np.array([[200,   0,  0,   0],
                  [0,   850,  0,  0],
                  [0,    0, 100,  0],
                  [0,    0,  0, 100]])
    R = np.array([[0.01]])

    profile, d_profile = generate_road_profile(dist, n_steps, Ts)

    # lqr = LQRController(suspension, Q=Q, R=R)
    # mpc = MPC(suspension, horizon=5, Q=Q, R=R)

    # suspension.simulate(n_steps, control_law=None,
    #                     reference=None, disturbance=d_profile)
    # suspension.plot_trajectory(axis=ax1, states=[1])

    # suspension.rst(x)
    # suspension.simulate(n_steps, control_law=lqr,
    #                     reference=None, disturbance=d_profile)
    # suspension.plot_trajectory(axis=ax1, states=[1])

    # suspension.rst(x)
    # suspension.simulate(n_steps, control_law=mpc,
    #                     reference=None, disturbance=d_profile)
    # suspension.plot_trajectory(axis=ax1, states=[1])

    λs_range = np.linspace(1, 5, 10)
    λg_range = np.linspace(1, 5, 10)

    iter_pairs = list(zip(range(λs_range.shape[0]), λs_range))

    pool = Pool(processes=4)

    sims = partial(sim_parellel, n_steps=n_steps, x=x, Ts=Ts,
                   d_profile=d_profile, λg_range=λg_range)

    result = pool.map(sims, iter_pairs)

    print(result)

    loss_map = np.zeros((λs_range.shape[0], λg_range.shape[0]), dtype=np.float64)
    
    for i, m in result:
        loss_map[i, :] = m

    loss_map = np.log10(loss_map)

    np.save("loss_map.npy", loss_map)

    # suspension.plot_trajectory(axis=ax1, states=[1])

    # ax1.plot(np.linspace(0, t_sim, n_steps), profile[:-1])
    # suspension.plot_control_input(axis=ax2)

    plt.matshow(loss_map)
    plt.show()


def sim_parellel(iter_pair, *, n_steps, x, Ts, d_profile, λg_range):
    i, λ_s = iter_pair[0], iter_pair[1]
    print(i, "Start")

    suspension = ActiveSuspension(x0=x, Ts=Ts)
    print(suspension)

    loss_map = np.zeros((1, λg_range.shape[0]), dtype=np.float64)

    for j, λ_g in enumerate(λg_range):

        excitation = OpenLoop.rnd_input(suspension, n_steps)
        dpc = DeePC(suspension, 4, 10, excitation, λ_s=λ_s, λ_g=λ_g)

        suspension.simulate(n_steps,
                            control_law=dpc,
                            reference=None,
                            disturbance=d_profile)

        loss_map[:, j] = dpc.get_total_loss()
    
    print(i, "Finished")
    return i, loss_map


if __name__ == "__main__":
    main()
    # map = np.load("loss_map.npy")
    # map = np.ma.array(map, mask=np.isnan(map))
    # plt.imshow(np.log10(map))
    # plt.show()
