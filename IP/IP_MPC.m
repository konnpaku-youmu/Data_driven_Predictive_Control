% point stabilization + Single shooting
clear all
close all
clc

clc;
clear;

import casadi.*
load setpoint.mat

%% constants
Rm = 2.6;
Km = 0.00767;
Kb = 0.00767;
Kg = 3.7;
M = 0.455;
l = 0.305;
m = 0.210;
r = 0.635e-2;
g = 9.81;

%% System model
A = [0        0                     1               0;
     0        0                     0               1;
     0     -(m*g)/M      -(Kg^2*Km*Kb)/(M*Rm*r^2)   0;
     0   (M+m)*g/(M*l)   (Kg^2*Km*Kb)/(M*Rm*r^2*l)  0];

B = [         0;
              0; 
       (Km*Kg)/(M*Rm*r);
      (-Km*Kg)/(r*Rm*M*l)];

C = [1 0 0 0;
     0 1 0 0];

D = [0;0];

T = 0.05; % sampling time [s]
N = 24; % prediction horizon

v_max = 5; v_min = -v_max;
omega_max = pi/4; omega_min = -omega_max;

x = SX.sym('x'); 
x_n = SX.sym('x_n'); 
alpha = SX.sym('alpha');
alpha_n = SX.sym('alpha_n');

states = [x;x_n;alpha;alpha_n]; 
n_states = length(states);

v = SX.sym('v'); 
controls = [v]; 
n_controls = length(controls);

rhs = A*states + B*controls; % system r.h.s

f = Function('f',{states,controls},{rhs}); % nonlinear mapping function f(x,u)
U = SX.sym('U',n_controls,N); % Decision variables (controls)
P = SX.sym('P',n_states + n_states);
% parameters (which include the initial and the reference state of the robot)

X = SX.sym('X',n_states,(N+1));
% A Matrix that represents the states over the optimization problem.

% compute solution symbolically
X(:,1) = P(1:4); % initial state
for k = 1:N
    st = X(:,k);  con = U(:,k);
    f_value  = f(st,con);
    st_next  = st+ (T*f_value);
    X(:,k+1) = st_next;
end
% this function to get the optimal trajectory knowing the optimal solution
ff=Function('ff',{U,P},{X});

obj = 0; % Objective function
g = [];  % constraints vector

Q = zeros(4,4); Q(1,1) = 0.5;Q(2,2) = 4;Q(3,3) = 0.1;Q(4,4) = 0.1; % weighing matrices (states)
R = 0.005; % weighing matrices (controls)
% compute objective
for k=1:N
    st = X(:,k);  con = U(:,k);
    obj = obj+(st-P(5:8))'*Q*(st-P(5:8)) + con'*R*con; % calculate obj
end

% compute constraints
for k = 1:N+1   % box constraints due to the map margins
    g = [g ; X(1,k)];   %state x
    g = [g ; X(2,k)];   %state y
end

% make the decision variables one column vector
OPT_variables = reshape(U,N,1);
nlp_prob = struct('f', obj, 'x', OPT_variables, 'g', g, 'p', P);

opts = struct;
opts.ipopt.max_iter = 100;
opts.ipopt.print_level =0;%0,3
opts.print_time = 0;
opts.ipopt.acceptable_tol =1e-8;
opts.ipopt.acceptable_obj_change_tol = 1e-6;

solver = nlpsol('solver', 'ipopt', nlp_prob,opts);


args = struct;
% inequality constraints (state constraints)
args.lbg = -2;  % lower bound of the states x and y
args.ubg = 2;   % upper bound of the states x and y 

% input constraints
args.lbx(1:2:N-1,1) = v_min; args.lbx(2:2:N,1)   = omega_min;
args.ubx(1:2:N-1,1) = v_max; args.ubx(2:2:N,1)   = omega_max;
%----------------------------------------------
% ALL OF THE ABOVE IS JUST A PROBLEM SETTING UP


% THE SIMULATION LOOP SHOULD START FROM HERE
%-------------------------------------------
t0 = 0;
x0 = [0.1 ; 0 ; 0; 0];    % initial condition.
xs = [sp(:, 1) , zeros(length(sp), 1) , sp(:, 2), zeros(length(sp), 1)]; % Reference posture.

xx(:,1) = x0; % xx contains the history of states
t(1) = t0;

u0 = zeros(N,1);  % two control inputs 

sim_tim = 200; % Maximum simulation time

% Start MPC
mpciter = 0;
xx1 = [];
u_cl=[];

% the main simulaton loop... it works as long as the error is greater
% than 10^-2 and the number of mpc steps is less than its maximum
% value.
main_loop = tic;
while(mpciter < sim_tim / T)
    args.p   = [x0;xs(4500+mpciter+1, :)']; % set the values of the parameters vector
    args.x0 = reshape(u0',N,1); % initial value of the optimization variables
    %tic
    sol = solver('x0', args.x0, 'lbx', args.lbx, 'ubx', args.ubx,...
            'lbg', args.lbg, 'ubg', args.ubg,'p',args.p);    
    %toc
    u = reshape(full(sol.x)',1,N)';
    ff_value = ff(u',args.p); % compute OPTIMAL solution TRAJECTORY
    xx1(:,1:4,mpciter+1)= full(ff_value)';
    
    u_cl= [u_cl ; u(1,:)];
    t(mpciter+1) = t0;
    [t0, x0, u0] = shift(T, t0, x0, u,f); % get the initialization of the next optimization step
    
    xx(:,mpciter+2) = x0;  
    mpciter
    mpciter = mpciter + 1;
end
main_loop_time = toc(main_loop);

ss_error = norm((x0-xs(mpciter+1, :)'),2);

plot(xs(4500:4500+mpciter, :))
hold on
plot(xx')

function [t0, x0, u0] = shift(T, t0, x0, u,f)
    st = x0;
    con = u(1,:)';
    f_value = f(st,con);
    st = st+ (T*f_value);
    x0 = full(st);
    
    t0 = t0 + T;
    u0 = [u(2:size(u,1),:);u(size(u,1),:)];
end
