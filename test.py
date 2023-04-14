import matplotlib.pyplot as plt
from Controller import OpenLoop, RndSetpoint, DeePC, CrappyPID
import numpy as np

from SysModels import IPNonlinear, Quadcopter, FlexJoint


def main():
    T_ini = 4
    pred_horizon = 10
    n_steps = 200

    x0 = np.array([[0], [0.2], [0], [0]])

    flex_joint = FlexJoint(Ts=0.05, x0=x0)
    # flex_joint.obsv()
    ctrl_open = OpenLoop.rnd_input(model=flex_joint, length=n_steps)

    Q = np.array([[10, 0],
                  [0,  2]])
    R = np.array([[0.02]])
    λ_s, λ_g = 15, 8
    deepc = DeePC(model=flex_joint, T_ini=T_ini,
                  horizon=pred_horizon, init_law=ctrl_open,
                  Q=Q, R=R, λ_s=λ_s, λ_g=λ_g)

    setpoint = RndSetpoint(flex_joint.n_outputs, n_steps, 
                           0, np.array([[[-2.5], [2.5]]]), 
                           switch_prob=0.02)
    flex_joint.simulate(n_steps, control_law=deepc, ref_traj=setpoint())

    flex_joint.plot_trajectory()
    deepc.plot_reference()
    plt.show()


if __name__ == "__main__":
    main()
 