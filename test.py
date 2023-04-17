import matplotlib.pyplot as plt
from Controller import OpenLoop, RndSetpoint, DeePC, CrappyPID
import numpy as np

from SysModels import IPNonlinear, Quadcopter, FlexJoint

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 12
})

def main():
    T_ini = 4
    pred_horizon = 10
    n_steps = 500

    x0 = np.array([[0], [0.2], [0], [0]])

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

    flex_joint = FlexJoint(Ts=0.05, x0=x0)
    # flex_joint.obsv()
    ctrl_open = OpenLoop.rnd_input(model=flex_joint, length=n_steps)

    Q = np.array([[12, 0],
                  [0,  4]])
    R = np.array([[0.01]])
    λ_s, λ_g = 15, 8
    deepc = DeePC(model=flex_joint, T_ini=T_ini,
                  horizon=pred_horizon, init_law=ctrl_open,
                  Q=Q, R=R, λ_s=λ_s, λ_g=λ_g)

    setpoint = RndSetpoint(flex_joint.n_outputs, n_steps, 
                           0, np.array([[[-2.5], [2.5]]]), 
                           switch_prob=0.02)
    flex_joint.simulate(n_steps, control_law=deepc, ref_traj=setpoint())

    flex_joint.plot_trajectory(ax1)
    deepc.plot_reference(ax1)
    deepc.plot_loss(ax2)
    plt.show()


if __name__ == "__main__":
    main()
 