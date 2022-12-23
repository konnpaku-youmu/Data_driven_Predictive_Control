import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.io as sio
from casadi import *
from casadi.tools import *
from typing import Tuple

# # Random seed:
# np.random.seed(1)


plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 14
})

from System import System

sys_dc = sio.loadmat('sys.mat') # discrete time model
Ad = sys_dc['A']
Bd = sys_dc['B']
C = sys_dc['C']
D = sys_dc['D']

x0 = np.array([[0.3], [0.2]])

sys = System(Ad, Bd, C, D, x0)

def random_u(u0, switch_prob=0.4, u_max=0.1):
    # Hold the current value with switch_prob chance or switch to new random value.
    u_next = (0.5-np.random.rand(u0.shape[0], 1))*u_max  # New candidate value.
    switch = np.random.rand() >= (1-switch_prob)  # switching? 0 or 1.
    u0 = (1-switch)*u0 + switch*u_next  # Old or new value.
    return u0

sys.reset(x0)

u0 = np.zeros((Bd.shape[1], 1))
for k in range(200):
    if k < 100:
        u0 = random_u(u0)
    else:
        u0 = np.zeros((Bd.shape[1], 1))
    sys.make_step(u0)

plt.figure(figsize=(6, 4))
plt.subplot(121)
plt.plot(sys.time, sys.y)
plt.legend(["dh1", "dh2"])
plt.subplot(122)
plt.step(sys.time, sys.u)
plt.legend(["dq1", "dq2"])

# Data collection
T_ini = 10
N = 40

L = T_ini + N

T = 150

n_u = sys.n_u
n_y = sys.n_y

U_L = []
Y_L = []

u0 = np.zeros((Bd.shape[1], 1))

for k in range(T):
    x0 = np.random.randn(Ad.shape[1], 1) * 0.2
    sys.reset(x0)

    for k in range(L):
        u0 = random_u(u0)
        sys.make_step(u0)

    U_L.append(sys.u.reshape(-1, 1))
    Y_L.append(sys.y.reshape(-1, 1))

print(Y_L[0].shape)

U_L = np.concatenate(U_L, axis=1)
Y_L = np.concatenate(Y_L, axis=1)

assert np.linalg.matrix_rank(U_L) == U_L.shape[0], "not persistantly exciting."

U_Tini, U_N = np.split(U_L, [n_u*T_ini], axis=0)
Y_Tini, Y_N = np.split(Y_L, [n_y*T_ini], axis=0)

print(Y_Tini.shape, Y_N.shape)  # 4, 40

# construct the optimizer
opt_x_dpc = struct_symMX([
    entry('g', shape=(T)),
    entry('u_N', shape=(n_u), repeat=N),
    entry('y_N', shape=(n_y), repeat=N)
])

opt_p_dpc = struct_symMX([
    entry('u_Tini', shape=(n_u), repeat=T_ini),
    entry('y_Tini', shape=(n_y), repeat=T_ini),
])

opt_x_num_dpc = opt_x_dpc(0)
opt_p_num_dpc = opt_p_dpc(0)


# Create the objective:
obj = 0
for k in range(N):
    obj += sum1(opt_x_dpc['y_N', k]**2)+0.1*sum1(opt_x_dpc['u_N', k]**2)



## Create the constraints
A = vertcat(Y_Tini, U_Tini, U_N, Y_N)
b = vertcat(*opt_p_dpc['y_Tini'], *opt_p_dpc['u_Tini'],
            *opt_x_dpc['u_N'], *opt_x_dpc['y_N'])

cons = A@opt_x_dpc['g']-b



# Create lower and upper bound structures and set all values to plus/minus infinity.
lbx_dpc = opt_x_dpc(-np.inf)
ubx_dpc = opt_x_dpc(np.inf)


# Set only bounds on u_N
lbx_dpc['u_N'] = -1e-2
ubx_dpc['u_N'] = 1e-2

# Create Optim
nlp = {'x': opt_x_dpc, 'f': obj, 'g': cons, 'p': opt_p_dpc}
S_dpc = nlpsol('S', 'ipopt', nlp)


# np.random.seed(10)
x0 = np.array([[-0.1], [-0.2]])
sys.reset(x0)

lbg = []
ubg = []

lb_states = np.array([-0.8, -0.4])
ub_states = np.array([0.2, 0.6])

# Excitement
n_exc = 100
u0 = np.zeros((Bd.shape[1], 1))
for k in range(n_exc):
    u0 = random_u(u0)
    sys.make_step(u0)
    lbg.append(lb_states)
    ubg.append(ub_states)

y_Tini = sys.y[-T_ini:, :]
u_Tini = sys.u[-T_ini:, :]

opt_p_num_dpc['y_Tini'] = vertsplit(y_Tini)
opt_p_num_dpc['u_Tini'] = vertsplit(u_Tini)


# solve the DPC problem
r = S_dpc(p=opt_p_num_dpc, lbg=0, ubg=0, lbx=lbx_dpc, ubx=ubx_dpc)
# Extract solution
opt_x_num_dpc.master = r['x']
u_N_dpc = horzcat(*opt_x_num_dpc['u_N']).full().T
y_N_dpc = horzcat(*opt_x_num_dpc['y_N']).full().T

# Plot the result
fig, ax = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

t = np.arange(N)*0.1
y_dpc_lines = ax[0].plot(t, y_N_dpc, linewidth=2)
ax[0].legend([r"$\Delta h_1$", r"$\Delta h_2$"])
ax[0].set_ylabel('Water level')
ax[0].set_prop_cycle(None)

u_dpc_lines = ax[1].step(t, u_N_dpc, linewidth=2)
ax[1].legend([r"$\Delta q_1$", r"$\Delta q_2$"])
ax[1].set_ylabel('Water flow')
ax[1].set_prop_cycle(None)

plt.show()
