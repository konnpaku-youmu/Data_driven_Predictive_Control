%SIMPLE LINEAR DEEPC EXAMPLE CODE

%IMPORTANT NOTE: THAT YOU NEED YALMIP TO RUN THIS SCRIPT DUE TO 
%THE WAY I'VE SETUP THE OPTIMIZATION PROBLEM

clear all
close all
clc

%% Variables to choose

control_horizon = 1; %choose how many time steps of optimal control to apply
N = 30;%prediction horizon
Tini = 2; %initial condition horizon
lambda_g = 500; %regularizer weight on g vector
lambda_y = 100000; %regularizer weight for slack variable

solver = 'quadprog'; %choose solver
options = sdpsettings('solver',solver,'verbose',0);

%SYSTEM
A = [1 1;0 1];
B = [0;1];
n = size(A,1);
m = size(B,2);
C = [1 0];
p = size(C,1);
D = zeros(p,m);

%Continuous linear system
ref_states = 1; %reference states
sys = ss(A, B, C, D);
K = -dlqr(A,B,eye(n),eye(m));

%% Choosing Data
min_num_data_pts = ((m+1)*(Tini+N+n)-1); %this is minimum number of data pts for persistency of excitation
num_data_pts = (m+p+1)*(Tini+N)+9;

%Collect data
x_data(:,1) = zeros(2,1);
for i = 1:num_data_pts
    u_data(i) = 10*randn(1,1);
    x_data(:,i+1) = A*x_data(:,i) + B*u_data(i);
    y_data(:,i) = C*x_data(:,i);
end

%Hankel matrices
H_u = data2hankel(u_data,Tini+N);
H_y = data2hankel(y_data,Tini+N);

U_p = H_u(1:m*Tini,:);
U_f = H_u(m*Tini+1:end,:);
Y_p = H_y(1:p*Tini,:);
Y_f = H_y(p*Tini+1:end,:);

%Build the u_data Hankel to check persistent excitation condition
H_u_pe = data2hankel(u_data,Tini+N+n);
if rank(H_u_pe) == size(H_u_pe,1)
    fprintf('Data Persistenly exciting of order Tini+N+n \n')
else
    fprintf('Data *NOT* Persistenly exciting\n')
end

tic

%% Get controllers

%Define constraint sets
x_min = -10*ones(n,1);
x_max = 10*ones(n,1);

state_constraints = [x_min x_max];

%SETUP DeePC OPTIMIZATION PROBLEM WITH YALMIP
uini = sdpvar(m*Tini, 1);
yini = sdpvar(p*Tini, 1);
r = sdpvar(length(ref_states),1);
sigma = sdpvar(p*Tini,1);
g = sdpvar(size(U_p,2),1);
U_f_times_g = U_f*g;
Y_f_times_g = Y_f*g;
u = reshape(U_f_times_g,m,N); %define control variable based on U_f*g
y = reshape(Y_f_times_g,p,N); %define output variable based on Y_f*g

constraints = repmat(C*state_constraints(:,1),1,N) <= y;
constraints = [constraints, y <= repmat(C*state_constraints(:,2),1,N)];
constraints = [constraints, (U_p*g == uini)];
constraints = [constraints, (Y_p*g == yini + sigma)];

ref_state_output_vector = reshape(y(ref_states,:),[],1);
u_vector = reshape(u,[],1);
objective = 200*norm(ref_state_output_vector-repmat(r,N,1),2)^2 + norm(u_vector,2)^2;
objective = objective + lambda_g*norm(g,2)^2 + lambda_y*norm(sigma,2)^2;

controller_DeePC = optimizer(constraints, objective, options, [uini; yini; r], u(:,1:control_horizon));

fprintf('Controller Computed\n')

%% Simulate on dynamics
sim_time = 100; %simulation time steps

%DeePC simulation
u_sim = zeros(m,sim_time);
x_sim = zeros(n,sim_time);
y_sim = zeros(p,sim_time);
sigma_sim = zeros(p*Tini,sim_time);
reference = zeros(p,sim_time);

%Initial states set to 0 for lack of a better choice
x_sim(:,1) = zeros(n,1);
uini_sim = zeros(m*Tini,1);
yini_sim = zeros(p*Tini,1);

for i = 1:sim_time
    r_sim = 6;%constant
    reference(:,i) = r_sim;
    u_sim(i) = controller_DeePC([uini_sim; yini_sim; r_sim]);
    x_sim(:,i+1) = A*x_sim(:,i) + B*u_sim(i);
    y_sim(:,i) = C*x_sim(:,i);
    uini_sim(1:m) = [];
    yini_sim(1:p) = [];
    uini_sim = [uini_sim; u_sim(i)];
    yini_sim = [yini_sim; y_sim(:,i)];
end

fprintf('Simulation complete\n')

toc


%% Plot results
t = 0:sim_time-1;

figure;
plot(t,y_sim(1,:),'Linewidth',2)
hold on
plot(t,reference(1,:),'--')
plot(t,repmat(state_constraints(1,1),length(t)),'--','Color','Red','Linewidth',2)
plot(t,repmat(state_constraints(1,2),length(t)),'--','Color','Red','Linewidth',2)
legend('Output','Reference','Constraints')
xlabel('Time')

%% FUNCTIONS
function H = data2hankel(data,num_block_rows)
%data =  (size of data entries) x (length of data stream)
%num_block_rows = number of block rows wanted in Hankel matrix
dim = size(data,1);
num_data_pts = size(data,2);
H = zeros(dim*(num_block_rows), num_data_pts-num_block_rows+1);

for i = 1:num_block_rows
    for j = 1:num_data_pts-num_block_rows+1
        H(dim*(i-1)+1:dim*i,j) = data(:,i+j-1);
    end
end

end