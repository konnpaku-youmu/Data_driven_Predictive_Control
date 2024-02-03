import matplotlib.pyplot as plt
from Controller import OpenLoop, RndSetpoint, DeePC, CrappyPID, LQRController, SMStruct
import numpy as np

from SysModels import IPNonlinear, Quadcopter, FlexJoint, InvertedPendulum


plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 12
})

def InvPen():
    T_ini = 4
    pred_horizon = 20
    n_steps = 200

    x0 = np.array([[0.8], [-0.2], [0], [0]])

    _, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

    InvPenNonlinear = IPNonlinear(Ts=0.05, x0=x0, plot_use=ax1)
    
    Q_ini = np.array([[3.5, 0, 0, 0],
                [0, 18, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0]])
    inv_pen_lin = InvertedPendulum(Ts=0.05, x0=x0, plot_use=ax1)
    init_lqr = LQRController(inv_pen_lin, Q=Q_ini, R=0.002)

    Q = np.array([[12, 0],
                  [0, 45]])
    R = np.array([[0.002]])
    λ_s, λ_g = 15, 8
    deepc = DeePC(model=InvPenNonlinear, T_ini=T_ini, data_mat=SMStruct.HANKEL,
                  horizon=pred_horizon, init_law=init_lqr,
                  Q=Q, R=R, λ_s=λ_s, λ_g=λ_g)

    setpoint = RndSetpoint(InvPenNonlinear.n_outputs, n_steps, 
                           0, np.array([[[-0.5], [0.5]]]), 
                           switch_prob=0.02)
    InvPenNonlinear.simulate(n_steps, control_law=deepc, ref_traj=setpoint())

    InvPenNonlinear.plot_trajectory()
    deepc.plot_reference(ax1)
    deepc.plot_loss(ax2)
    plt.show()

def main():
    T_ini = 4
    pred_horizon = 10
    n_steps = 200

    x0 = np.array([[0], [0.2], [0], [0]])

    _, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

    flex_joint = FlexJoint(Ts=0.05, x0=x0, plot_use=ax1)
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

    flex_joint.plot_trajectory()
    deepc.plot_reference(ax1)
    deepc.plot_loss(ax2)
    plt.show()


if __name__ == "__main__":
    main()