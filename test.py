import matplotlib.pyplot as plt
from Controller import LQRController, OpenLoop, SetpointGenerator, DeePC, CrappyPID
import numpy as np

from SysModels import SimpleHarmonic, InvertedPendulum, IPNonlinear, Quadcopter, FlexJoint

def main():
    T_ini = 4
    pred_horizon = 20
    n_steps = 200

    x0 = np.array([[0], [0.2], [0], [0]])
    flex_joint = FlexJoint(Ts=0.05, x0=x0)
    ctrl_open = OpenLoop.rnd_input(model=flex_joint, length=n_steps)

    deepc = DeePC(model=flex_joint, T_ini=T_ini, horizon=pred_horizon, 
                  init_law=ctrl_open)


    flex_joint.simulate(n_steps, control_law=deepc)

    flex_joint.plot_trajectory()
    plt.show()

if __name__ == "__main__":
    main()
