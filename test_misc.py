import matplotlib.pyplot as plt
from Controller import OpenLoop, RndSetpoint, DeePC, CrappyPID
import numpy as np

from SysModels import IPNonlinear, Quadcopter, FlexJoint
from helper import hankelize

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 12
})

def main():
    T_ini = 4
    pred_horizon = 10
    n_steps = 50

    

if __name__ == "__main__":
    main()
 