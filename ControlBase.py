import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from typing import Callable, List
from dataclasses import dataclass, field

import casadi as cs
from casadi.tools import *

from SysModels import System, LinearSystem, NonlinearSystem
from helper import hankelize, pagerize, RndSetpoint, Bound, ControllerType, SMStruct, OCPType
from rcracers.utils.geometry import Polyhedron, Ellipsoid


@dataclass
class InvSetResults:
    n_iter: int = 0
    iterations: List[Polyhedron] = field(default_factory=list)
    success: bool = False

    def increment(self):
        self.n_iter += 1

    @property
    def solution(self) -> Polyhedron:
        return self.iterations[-1]

class Controller:
    controller_type: ControllerType = None

    def __init__(self, model: System, **kwargs) -> None:
        self.model = model
        self.closed_loop = None
        self.u = None

    def __call__(self, x: np.ndarray, r: np.ndarray) -> np.ndarray:
        raise NotImplementedError()


class OpenLoop(Controller):
    controller_type = ControllerType.VANILLA

    def __init__(self, model: System) -> None:
        super().__init__(model)
        self.closed_loop = False

    @classmethod
    def given_input_seq(cls, model: System, u: np.ndarray):
        inst = cls.__new__(cls)
        super(OpenLoop, inst).__init__(model=model)
        inst.__set_input_sequence(u=u)
        return inst

    @classmethod
    def rnd_input(cls, model: System, length: int,
                  lb: np.ndarray = None, ub: np.ndarray = None):
        inst = cls.__new__(cls)
        super(OpenLoop, inst).__init__(model=model)

        if lb is None:
            lb = model.input_constraint.lb

        if ub is None:
            ub = model.input_constraint.ub

        inst.__rnd_input_seq(length=length,
                             lbu=lb,
                             ubu=ub)
        return inst

    def __set_input_sequence(self, u: np.ndarray) -> None:
        self.u = np.atleast_3d(u)

    def __rnd_input_seq(self, length: int, lbu: np.ndarray, ubu: np.ndarray, switch_prob: float = 0.8) -> None:
        assert (lbu.shape == ubu.shape)

        self.u = np.zeros([length, lbu.shape[0], ubu.shape[1]])

        for k in range(length):
            if np.random.rand() <= switch_prob:
                self.u[k] = np.random.uniform(lbu*0.1, ubu*0.1, self.u.shape[1:])
            else:
                self.u[k] = self.u[k-1]

    def __call__(self, x: np.ndarray, k: int) -> np.ndarray:
        u = self.u[0]
        self.u = np.roll(self.u, -1, axis=0)
        return u, None


class LQRController(Controller):
    controller_type = ControllerType.VANILLA

    def __init__(self, model: LinearSystem, **kwargs) -> None:
        super().__init__(model)

        self.closed_loop = True
        self.K = None

        try:
            self.Q = kwargs["Q"]
        except KeyError:
            C = model.C
            ns = model.n
            self.Q = C.T@C + np.eye(ns, ns) * 1e-2

        try:
            self.R = kwargs["R"]
        except KeyError:
            nu = model.m
            self.R = np.eye(nu, nu) * 0.1

        self.compute_K()

    def set_weights(self, Q: np.ndarray, R: np.ndarray) -> None:
        self.Q, self.R = Q, R
        self.compute_K()

    def compute_K(self) -> None:
        A = self.model.A
        B = self.model.B

        P = linalg.solve_discrete_are(A, B, self.Q, self.R)
        self.K = -np.linalg.inv(self.R + B.T@P@B)@B.T@P@A

    def __call__(self, x: np.ndarray, r: int) -> np.ndarray:
        u = self.K@(x - r)

        if u >= self.model.input_constraint.ub:
            u = self.model.input_constraint.ub
        elif u <= self.model.input_constraint.lb:
            u = self.model.input_constraint.lb

        return u, None
