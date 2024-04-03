import numpy as np

import casadi as cs
from casadi.tools import *

from SysModels import System
from helper import Bound, ControllerType

from ControlBase import Controller

class MPC(Controller):
    controller_type = ControllerType.PREDICTIVE
    name = "MPC"

    def __init__(self, model: System,
                 horizon: int,
                 *,
                 enforce_term: bool = False,
                 **kwargs) -> None:

        super().__init__(model)

        kwargs.setdefault("state_bound", self.model.state_constraint)
        kwargs.setdefault("output_bound", self.model.output_constraint)
        kwargs.setdefault("input_bound", self.model.input_constraint)

        kwargs.setdefault("Q", np.eye(self.model.p, self.model.p) * 10)
        kwargs.setdefault("R", np.eye(self.model.m, self.model.m) * 1e-2)
        kwargs.setdefault("Pf", np.eye(self.model.p, self.model.p) * 10)

        self.closed_loop = True

        self.horizon = horizon

        self.Q = kwargs["Q"]
        self.R = kwargs["R"]
        self.Pf = kwargs["Pf"]

        self.enforce_term = enforce_term

        self.problem = None
        self.solver = None

        self.state_bound: Bound = kwargs["state_bound"]
        self.output_bound: Bound = kwargs["output_bound"]
        self.input_bound: Bound = kwargs["input_bound"]

        self.opt_params: struct_symMX = None
        self.opt_vars: struct_symMX = None
        self.param_vals = None
        self.var_vals = None
        self.ref = None    # Tracking reference

        self.solver = None
        self.bounds = None

        self.objective = []
    
    def build(self) -> None:
        self.solver, self.bounds = self.__build_mpc_problem()
        return

    def __build_mpc_problem(self) -> None:
        print("Building MPC problem ...")

        lb_states = self.state_bound.lb
        ub_states = self.state_bound.ub
        lb_output = self.output_bound.lb
        ub_output = self.output_bound.ub
        lb_inputs = self.input_bound.lb
        ub_inputs = self.input_bound.ub

        state_cons, output_cons = [], []
        lbu, ubu, lbx, ubx = [], [], [], []
        cost = 0

        Q, R, Pf = self.Q, self.R, self.Pf

        opt_params = struct_symMX([entry('x0', shape=(self.model.n, 1)),
                                   entry('ref', shape=(self.model.p, 1))])

        opt_vars = struct_symMX([entry("u", shape=(self.model.m), repeat=self.horizon)])

        self.param_vals = opt_params(0)
        self.var_vals = opt_vars(0)

        x = opt_params["x0"]

        for k in range(self.horizon):
            uk = opt_vars["u", k]
            wk = np.zeros((self.model.m2, 1))

            y = self.model._output(x, uk) - opt_params['ref']

            cost += y.T@Q@y + uk.T@R@uk

            x = self.model._f(x0=x, p=uk, w=wk)

            lbu.append(lb_inputs)
            ubu.append(ub_inputs)

            output_cons.append(y)
            lbx.append(lb_output)
            ubx.append(ub_output)

            output_cons.append(x)
            lbx.append(lb_states)
            ubx.append(ub_states)

        y = self.model._output(x, np.zeros((self.model.m, 1)))
        cost += y.T@Pf@y

        self.problem = {"x": opt_vars,  # optimized variables: input u
                        "f": cost,
                        "g": cs.vertcat(*output_cons),
                        "p": opt_params}

        opts = {"ipopt.tol": 1e-12,
                "ipopt.max_iter": 200,
                "ipopt.print_level": 0,
                "expand": True,
                "verbose": False,
                "print_time": False}

        solver = nlpsol("solver", "ipopt", self.problem, opts)

        bounds = dict(
            lbx=cs.vertcat(*lbu), ubx=cs.vertcat(*ubu),
            lbg=cs.vertcat(*lbx), ubg=cs.vertcat(*ubx)
        )

        return solver, bounds

    def __update_ref(self, r: np.ndarray) -> None:
        if self.ref is None:
            self.ref = np.zeros([1, r.shape[0], r.shape[1]])
            self.ref[0, :, :] = r
        else:
            self.ref = np.concatenate([self.ref,
                                       np.atleast_3d(r.squeeze())], axis=0)

    def __call__(self, x: np.ndarray, r: np.ndarray) -> np.ndarray:
        self.param_vals['x0'] = x
        self.param_vals['ref'] = vertsplit(r.squeeze())
        self.__update_ref(r)

        res = self.solver(p=self.param_vals, **self.bounds)

        # Extract optimal solution
        self.var_vals.master = res['x']
        loss_val = res['f']
        opti_u = self.var_vals['u']

        self.objective.append(loss_val)

        return opti_u[0], opti_u[self.model.m:]

    def plot_loss(self, axis=None, **pltargs) -> None:
        pltargs.setdefault("linewidth", 1)

        y = self.model.get_y()
        length = y.shape[0]
        plot_range = np.linspace(
            0, length*self.model.Ts, length, endpoint=False)

        axis.semilogy(plot_range, np.array(self.objective)[
                      :, 0, 0], label=r"$f$:{{{0}}}, $N = {{{1}}}$".format(self.name, self.horizon))
        axis.set_xlim(0,  y.shape[0]*self.model.Ts)
        axis.legend()


class MPFC(MPC):
    def __init__(self, model: System, horizon: int, *, enforce_term: bool = False, **kwargs) -> None:
        super().__init__(model, horizon, enforce_term=enforce_term, **kwargs)

        self.contour_dim = 2
    
    def build(self) -> None:
        self.solver, self.bounds = self.__build_mpc_problem()
        return

    def __build_mpc_problem(self) -> None:
        print("Building MPFC problem ...")
        x0 = cs.SX.sym(f"x0", (self.model.n, 1))
        x = x0

        lb_states = self.state_bound.lb
        ub_states = self.state_bound.ub
        lb_output = self.output_bound.lb
        ub_output = self.output_bound.ub
        lb_inputs = self.input_bound.lb
        ub_inputs = self.input_bound.ub

        state_cons, output_cons = [], []
        lbu, ubu, lbx, ubx = [], [], [], []
        cost = 0

        Q, R, Pf = self.Q, self.R, self.Pf

        self.opt_params = struct_symSX([entry('x0', shape=(self.model.n, 1)),
                                   entry('ref', shape=(self.contour_dim, 1), repeat=self.horizon)])

        self.opt_vars = struct_symSX([entry("u", shape=(self.model.m), repeat=self.horizon)])

        x = self.opt_params["x0"]

        for k in range(self.horizon):
            uk = self.opt_vars["u", k]
            wk = np.zeros((self.model.m2, 1))

            y = self.model._output(x, uk)
            err_t = y[:self.contour_dim] - self.opt_params['ref', k]

            cost += err_t.T@Q@err_t + uk.T@R@uk

            x = self.model._f(x0=x, p=uk, w=wk)

            lbu.append(lb_inputs)
            ubu.append(ub_inputs)

            output_cons.append(y)
            lbx.append(lb_output)
            ubx.append(ub_output)

            output_cons.append(x)
            lbx.append(lb_states)
            ubx.append(ub_states)

        y = self.model._output(x, np.zeros((self.model.m, 1)))
        err_t = y[:self.contour_dim]- self.opt_params['ref', self.horizon-1]
        cost += err_t.T@Pf@err_t

        self.problem = {"x": self.opt_vars,  # optimized variables: input u
                        "f": cost,
                        "g": cs.vertcat(*output_cons),
                        "p": self.opt_params}

        opts = {"ipopt.tol": 1e-12,
                "ipopt.max_iter": 200,
                "ipopt.print_level": 0,
                "expand": True,
                "verbose": False,
                "print_time": False}

        solver = nlpsol("solver", "ipopt", self.problem, opts)

        bounds = dict(
            lbx=cs.vertcat(*lbu), ubx=cs.vertcat(*ubu),
            lbg=cs.vertcat(*lbx), ubg=cs.vertcat(*ubx)
        )

        return solver, bounds

    def __call__(self, x: np.ndarray, r: np.ndarray) -> np.ndarray:
        param_vals, var_vals = self.opt_params(0), self.opt_vars(0)
        
        param_vals['x0'] = x
        param_vals['ref'] = vertsplit(r.squeeze())

        res = self.solver(p=param_vals, **self.bounds)

        # Extract optimal solution
        var_vals.master = res['x']
        loss_val = res['f']
        opti_u = var_vals['u']

        self.objective.append(loss_val)

        return opti_u[0], opti_u[self.model.m:]
    

class MPCC(MPC):
    def __init__(self, model: System, horizon: int, *, enforce_term: bool = False, **kwargs) -> None:
        super().__init__(model, horizon, enforce_term=enforce_term, **kwargs)
        self.contour_dim = 2
    
    def build(self):
        self.solver, self.bounds = self.__build_mpc_problem()
        return

    def __build_mpc_problem(self) -> None:

        lb_states = self.state_bound.lb
        ub_states = self.state_bound.ub
        lb_output = self.output_bound.lb
        ub_output = self.output_bound.ub
        lb_inputs = self.input_bound.lb
        ub_inputs = self.input_bound.ub

        self.opt_params = struct_symSX([entry("x0", shape=(self.model.n, 1)),
                                        entry("ref", shape=(self.model.p, 1))])
        
        self.opt_vars = struct_symSX()

        x = self.opt_params["x0"]



        return None, None
    
    def __loss(self) -> None:
        
        return