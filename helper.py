import numpy as np
from enum import Enum
from typing import Tuple, Callable
from dataclasses import dataclass
from scipy import linalg
import casadi as cs
import matplotlib.pyplot as plt

from xml.dom import minidom
from svgpathtools import parse_path

from casadi import *
# def forward_euler(A: np.ndarray, B: np.ndarray, Ts: float) -> Tuple[np.ndarray]:
#     n_states = A.shape[1]

#     Ad = np.eye(n_states) + Ts * A
#     Bd = Ts * B

#     return Ad, Bd

def setup_plot():
    fig1 = plt.figure(figsize=(14, 5))
    ax1 = fig1.add_subplot(1, 2, 1)
    ax2 = fig1.add_subplot(1, 2, 2)
    fig1.tight_layout()

    return ax1, ax2

def forward_euler(f, Ts) -> Callable:
    def fw_eul(x0, p, w):
        return x0 + f(x0, p, w) * Ts
    return fw_eul


def zoh(A: np.ndarray, B: np.ndarray, Ts: float) -> Tuple[np.ndarray]:
    em_upper = np.hstack((A, B))

    # Need to stack zeros under the a and b matrices
    em_lower = np.hstack((np.zeros((B.shape[1], A.shape[0])),
                          np.zeros((B.shape[1], B.shape[1]))))

    em = np.vstack((em_upper, em_lower))

    ms = linalg.expm(Ts * em)

    # Dispose of the lower rows
    ms = ms[:A.shape[0], :]

    Ad = ms[:, 0:A.shape[1]]
    Bd = ms[:, A.shape[1]:]

    return Ad, Bd


def rk4(f: cs.Function, Ts: float) -> Callable:
    def rk4_dyn(x0, p, w):
        s_1 = f(x0, p, w)
        s_2 = f(x0 + (Ts / 2) * s_1, p, w)
        s_3 = f(x0 + (Ts / 2) * s_2, p, w)
        s_4 = f(x0 + Ts * s_3, p, w)
        x_next = x0 + (Ts / 6) * (s_1 + 2*s_2 + 2*s_3 + s_4)
        return x_next

    return rk4_dyn


def hankelize(vec: np.ndarray, L: int) -> np.ndarray:
    T = vec.shape[0]
    n = vec.shape[1]
    assert (T >= L)

    H = np.zeros([L*n, T-L+1])

    for i in range(T-L+1):
        H[:, i] = vec[i:i+L, :, :].reshape([L*n])

    return H


def pagerize(vec: np.ndarray, L: int, S: int = None) -> np.ndarray:
    N = vec.shape[0]
    n = vec.shape[1]
    if S > L:
        print("Stride larger than L. Reset to L")
        S = L

    k = (N-L)//S + 1

    if (N-L) % S != 0:
        N = (k-1) * S + L
        vec = vec[:N]

    P = np.zeros([L*n, k])

    for i in range(k):
        P[:, i] = vec[i*S:i*S+L, :, :].reshape([L*n])

    return P


def generate_road_profile(length: int, samples: int, Ts: float, type: str = "step"):

    d = np.linspace(5, length+5, samples+1)
    profile = np.zeros_like(d)

    if type == "step":
        pos = int(samples / 4)
        profile[pos:] = 0.1  # A 5cm high step
    elif type == "bump":
        ...
    elif type == "wave":
        profile = np.maximum(0.1*np.sin(0.05*np.pi*d), 0)  # Rectified sine wave, height = 10cm

    # differentiate the profile
    d_profile = np.array([(profile[i] - profile[i-1])/Ts for i in range(1, samples+1)])
    d_profile = np.atleast_3d(d_profile.squeeze()).reshape([samples, -1, 1])

    return profile, d_profile


@dataclass
class Bound:
    lb: np.ndarray = None
    ub: np.ndarray = None


class ControllerType(Enum):
    VANILLA = 0
    PREDICTIVE = 1


class SMStruct(Enum):
    HANKEL = 0
    PARTIAL_HANKEL = 1
    PAGE = 2


class OCPType(Enum):
    CANONICAL = 0
    REGULARIZED = 1


class RndSetpoint:
    def __init__(self, n_output, n_steps, trac_states: list,
                 bounds: np.ndarray, **kwargs) -> None:

        kwargs.setdefault("switch_prob", 0.05)
        switching_prob = kwargs["switch_prob"]

        self.sp = np.zeros([n_steps, n_output, 1])

        if isinstance(trac_states, int):
            trac_states = [trac_states]

        assert bounds.shape[0] == len(trac_states)

        for state, bound in zip(trac_states, bounds):
            sp_state = np.zeros([n_steps, 1])

            for k in range(n_steps):
                if np.random.rand() <= switching_prob:
                    sp_state[k] = np.random.uniform(np.min(bound), np.max(bound))
                else:
                    sp_state[k] = sp_state[k-1]

            self.sp[:, state, :] = sp_state

    def __call__(self) -> np.ndarray:
        return self.sp


class Plotter:

    def __init__(self) -> None:
        ...


class Track(object):
    def __init__(self, svg_file: str, density: int = 100) -> None:

        self.traj = self.__parse_svg(svg_file)
        self.nsteps = density
        self.step = 0
        self.horizon = 15

    def __parse_svg(self, svg_file: str):
        path = minidom.parse(svg_file)
        tag = path.getElementsByTagName("path")
        d_string = tag[0].attributes['d'].value

        path = parse_path(d_string)

        return path.scaled(0.152, 0.152)

    def __call__(self):
        
        pts = np.ndarray(shape=[self.horizon, 2, 1])

        step = self.step

        for i in range(self.horizon):
            progress = (step / self.nsteps) % 1.0
            pt = self.traj.point(progress)
            pt = np.array([[np.real(pt)],
                         [np.imag(pt)]])
            pts[i, :, :] = pt
            step += 1
        
        self.step += 1

        return pts

    def plot_traj(self, axis: plt.Axes):
        tval = np.linspace(0, 1, self.nsteps, endpoint=False)
        pts = np.ndarray(shape=[self.nsteps, 2, 1])
        for i, t in enumerate(tval):
            pt = self.traj.point(t)
            pt = np.array([[np.real(pt)],
                           [np.imag(pt)]])
            pts[i, :, :] = pt
        axis.plot(pts[:, 0, :], pts[:, 1, :], linestyle="--", color="#7e7e7e")
        return


if __name__ == "__main__":
    z = SX.sym('z', 2)
    x = SX.sym('x', 2)
    g0 = sin(x+z)
    g1 = cos(x-z)
    g = Function('g', [z, x], [g0, g1])
    G = rootfinder('G', 'newton', g)
    print(np.array(G(1, 1)))
