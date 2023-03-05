import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.io as sio
from casadi import *
from casadi.tools import *
import time
from System import System, random_u

# Random seed:
np.random.seed(1234)

# Customizing Matplotlib:
mpl.rcParams['font.size'] = 18
mpl.rcParams['lines.linewidth'] = 3
mpl.rcParams['axes.grid'] = True
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams['axes.unicode_minus'] = 'true'
mpl.rcParams['axes.labelsize'] = 'large'
mpl.rcParams['legend.fontsize'] = 'large'
mpl.rcParams['xtick.labelsize'] = 'large'
mpl.rcParams['ytick.labelsize'] = 'large'
mpl.rcParams['axes.labelpad'] = 6

sys_dc = sio.loadmat('/home/yz/Projects/Data_driven_Predictive_Control/toy_example/sys_dc.mat')
A = sys_dc['A_dc']
B = sys_dc['B_dc']
C = sys_dc['C']
D = sys_dc['D']

sys = System(A,B,C,D)

T_ini = 4
N = 40

L = T_ini + N

n_u = sys.n_u
n_y = sys.n_y

sig_w = 1e-2

def get_data_matrices(T, sig_w):
    U_L = []
    Y_L = []
    
    sys = System(A,B,C,D)

    u0 = np.zeros((2,1))

    for k in range(T):
        x0 = np.random.randn(8,1)
        sys.reset(x0)


        for k in range(L):
            u0 = random_u(u0)
            sys.make_step(u0)

        U_L.append(sys.u.reshape(-1,1))
        Y_L.append(sys.y.reshape(-1,1))

    U_L = np.concatenate(U_L,axis=1)
    Y_L = np.concatenate(Y_L,axis=1)

    # Add noise to Data:
    Y_L = Y_L+np.random.randn(*Y_L.shape)*sig_w

    assert np.linalg.matrix_rank(U_L) == U_L.shape[0], "not persistantly exciting."
    
    return U_L, Y_L

def setup_DeePC(U_L, Y_L, T):
    # Configure data
    U_Tini, U_N = np.split(U_L, [n_u*T_ini],axis=0)
    Y_Tini, Y_N = np.split(Y_L, [n_y*T_ini],axis=0)
    M = np.concatenate((Y_Tini, U_Tini, U_N))
    
    # Define optim variables
    opt_x_dpc = struct_symMX([
    entry('g', shape=(T)),
    entry('u_N', shape=(n_u), repeat=N),
    entry('y_N', shape=(n_y), repeat=N),
    entry('sig_y', shape=(n_y), repeat=T_ini),
    entry('sig_u', shape=(n_u), repeat=T_ini)
    ])

    opt_p_dpc = struct_symMX([
        entry('u_Tini', shape=(n_u), repeat=T_ini),
        entry('y_Tini', shape=(n_y), repeat=T_ini),
        entry('lam_g'),
    ])

    opt_x_num_dpc = opt_x_dpc(0)
    opt_p_num_dpc = opt_p_dpc(0)
    

    obj = 0
    for k in range(N):
        obj += sum1(opt_x_dpc['y_N',k]**2)+0.1*sum1(opt_x_dpc['u_N', k]**2)
    
    for k in range(T_ini):
        obj += 1e4*sum1(opt_x_dpc['sig_u',k]**2)+1e4*sum1(opt_x_dpc['sig_y',k]**2)
    
    obj += opt_p_dpc['lam_g']*sum1(opt_x_dpc['g']**2)


    # Create the constraints:
    b = vertcat(*opt_p_dpc['y_Tini'], *opt_p_dpc['u_Tini'], DM.zeros((N*n_u,1)))
    v = vertcat(*opt_x_dpc['sig_y'], *opt_x_dpc['sig_u'], *opt_x_dpc['u_N'])
    y_N = vertcat(*opt_x_dpc['y_N'])
    g = opt_x_dpc['g']
    
    cons = vertcat(
        M@g-b-v,
        Y_N@g-y_N
    )
    
    # Create lower and upper bound structures and set all values to plus/minus infinity.
    lbx = opt_x_dpc(-np.inf)
    ubx = opt_x_dpc(np.inf)


    # Set only bounds on u_N
    lbx['u_N'] = -0.7
    ubx['u_N'] = 0.7

    # Create Optim
    nlp = {'x':opt_x_dpc, 'f':obj, 'g':cons, 'p':opt_p_dpc}
    S_dpc = nlpsol('S', 'ipopt', nlp)
    
    return S_dpc, opt_x_num_dpc, opt_p_num_dpc, lbx, ubx



T_arr = [100, 150, 200]
N_sim = 60
N_exp = 10

## capture
res_deePC = []
for T in T_arr:
        
    # Repeat experiment 10 times.
    res = []
    for i in range(N_exp):
        np.random.seed(12)
        sys.reset(x0=np.zeros((8,1)))
        # Excitement
        n_exc = 20
        u0 = np.zeros((2,1))
        for k in range(n_exc):
            u0 = random_u(u0)
            sys.make_step(u0)
        
        for p in range(i):
            # Restore randomness.
            np.random.randn()

        U_L, Y_L = get_data_matrices(T, sig_w)

        S_dpc, opt_x_num_dpc, opt_p_num_dpc, lbx, ubx = setup_DeePC(U_L,Y_L,T)
        
        opt_p_num_dpc['lam_g'] = 1

        cost = []
        t_calc = []
        for k in range(N_sim):

            y_Tini = sys.y[-T_ini:,:]+sig_w*np.random.randn(T_ini,n_y)
            u_Tini = sys.u[-T_ini:,:]

            opt_p_num_dpc['y_Tini'] = vertsplit(y_Tini)
            opt_p_num_dpc['u_Tini'] = vertsplit(u_Tini)

            tic = time.time()
            r = S_dpc(p=opt_p_num_dpc, lbg=0, ubg=0, lbx=lbx, ubx=ubx)
            toc = time.time()

            opt_x_num_dpc.master = r['x']    
        
            u0 = opt_x_num_dpc['u_N',0].full().reshape(-1,1)
            y0 = sys.make_step(u0)

            cost.append(.1*u0.T@u0+y0.T@y0)
            t_calc.append(toc-tic)


        res.append({'time':sys.time[n_exc:], 'u':sys.u[n_exc:], 'y':sys.y[n_exc:], 'cost': np.concatenate(cost),'t_calc': np.array(t_calc)})
        
    res_deePC.append(res)



result_summary_deePC= {'mean_cost':[],'std_cost':[], 'mean_t_calc':[], 'std_t_calc':[]}

for res_i in res_deePC:
    cost = [np.sum(res_ik['cost']) for res_ik in res_i]
    t_calc = [np.mean(res_ik['t_calc']) for res_ik in res_i]
    result_summary_deePC['mean_cost'].append(np.round(np.mean(cost),3))
    result_summary_deePC['std_cost'].append(np.round(np.std(cost),3))
    result_summary_deePC['mean_t_calc'].append(np.round(np.mean(t_calc)*1e3,3))
    result_summary_deePC['std_t_calc'].append(np.round(np.std(t_calc)*1e3,3))
    
result_summary_deePC



fig, ax = plt.subplots(2,2,sharex=True, sharey=True, figsize=(10,6))

T_plot = 0

ax[0,0].plot(res_deePC[T_plot][0]['y'])
ax[1,0].plot(res_deePC[T_plot][0]['u'])

ax[0,0].set_title('DeePC')

plt.show()
