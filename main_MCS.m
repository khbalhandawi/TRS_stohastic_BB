
clearvars
close all
clc
addpath SGTE_matlab_server
addpath Support_functions

%% Problem definition
% Model parameters
mesh_size = 20;
mesh_size_AM = 10.0;
T_melt = 1500;
H_subs = 1e-5;
T_ref = 25.0;

IP_nominal = 0.0; % Nominal pressure load MPa

laser_power =  3806.054973;% Laser power
b_thick = 5.0; % TRS base thickness

stiff_thick = 20; % thickness of stiffener ribs
scanning_speed = 5.0; % scannning speed
power_density = 6.0; % power_density

% Model variables
bounds = [45.0           , 155.0                    ;... % Axial Position #1
          2.0            , 20                       ;... % Stiff height   #2
          20.0           , 155.0                    ;... % Stiff width    #3
          3500           , 4500                     ;... % Laser power    #4
          -100.0         , 100.0                    ;... % T1             #5
          -100.0         , 100.0                    ;... % T2             #6
          -100.0         , 100.0                    ;... % T3             #7
          -100.0         , 100.0                    ;... % T4             #8
          190.0          , 250.0                    ];   % Shroud width   #9

load('DOE_V1.mat','ax_pos','st_height','st_width','laser_power','T1_n','T2_n','T3_n','T4_n','shroud_width','N_th','n_f_th');
lb = bounds(:,1)'; ub = bounds(:,2)';
lob_v = bounds(1:3,1)'; upb_v = bounds(1:3,2)';
lob_p = bounds(4:end,1)'; upb_p = bounds(4:end,2)';

lhs_data = [ax_pos, st_height, st_width, laser_power, T1_n, T2_n, T3_n, T4_n, shroud_width];
lhs_data_normalize = scaling(lhs_data,lb,ub,1);

obj_data = [n_f_th]';

%% Construct surrogate models <---------------------------------------------------- CHOOSE DIFFERENT SURROGATE MODELS
%-------------------------------------------------------------------------%
% For default hyperparameters
% model = 'TYPE LOWESS DEGREE 2 KERNEL_TYPE D1 KERNEL_SHAPE 1.12073 DISTANCE_TYPE NORM2 RIDGE 0.0125395';
% model = 'TYPE KRIGING RIDGE 1.01723e-16 DISTANCE_TYPE NORM2 METRIC OECV BUDGET 200';
% For Hyperparameter Optimization
budget = 200; out_file = 'surrogate_model.sgt';
% model = ['TYPE LOWESS ', 'DEGREE OPTIM ', 'RIDGE OPTIM ', 'KERNEL_TYPE OPTIM ', 'KERNEL_COEF OPTIM ', 'DISTANCE_TYPE OPTIM ', 'METRIC OECV ', 'BUDGET ', num2str(budget), ' OUTPUT ', out_file];
% model = ['TYPE KS ', 'KERNEL_TYPE OPTIM ', 'KERNEL_COEF OPTIM ', 'DISTANCE_TYPE OPTIM ', 'METRIC OECV ','BUDGET ', num2str(budget), ' OUTPUT ', out_file];
% model = ['TYPE RBF ', 'KERNEL_TYPE OPTIM ', 'KERNEL_COEF OPTIM ', 'DISTANCE_TYPE OPTIM ', 'RIDGE OPTIM ', 'METRIC OECV ', 'BUDGET ', num2str(budget), ' OUTPUT ', out_file];
% model = ['TYPE KRIGING ', 'RIDGE OPTIM ', 'DISTANCE_TYPE OPTIM ', 'METRIC OECV ', 'BUDGET ', num2str(budget), ' OUTPUT ', out_file];
model = ['TYPE ENSEMBLE ', 'WEIGHT OPTIM ', 'METRIC OECV ', 'DISTANCE_TYPE OPTIM ','BUDGET ', num2str(budget),' OUTPUT ', out_file];
%-------------------------------------------------------------------------%

sgtelib_server_start(model,true,true)
% Test if server is ok and ready
sgtelib_server_ping;
% Feed server
sgtelib_server_newdata(lhs_data_normalize,obj_data');

%% Optimization problem in NOMAD
global index

lb_n = zeros(size(lob_v)); % lower bounds for NOMAD
ub_n = ones(size(lob_v)); % upper bounds for NOMAD

x0 = 0.5 * ones(size(lob_v));  % <--------------------------------------------------------- SET INITIAL GUESS HERE

index = 0; % Initialize counter
param = {index,lob_v,upb_v,lob_p,upb_p,T_melt};

%% Blackbox call
n_var = 3; n_samples = 1000; n_steps = 5;
var_DOE = linspace(0,1,n_steps);

for n = 1:1:n_steps
    
    DOE_filename = ['DOE_R',num2str(n),'.log']; % Purge out SBCE log file
    fileID_run = fopen(['MCS_results/',DOE_filename],'w');
    fclose(fileID_run);
    
    x0(n_var) = var_DOE(n);
    
    MCS_runs = zeros(n_samples,1);
    for m = 1:1:n_samples
        f = TRS_BB(x0,param);
        MCS_runs(m) = f(1);
        
        fileID_run = fopen(['MCS_results/',DOE_filename],'a');
        Net_results = sprintf('%f,' , [x0 f]);
        Net_results = Net_results(1:end-1);% strip final comma
        fprintf(fileID_run, '%i,%s\n', [m,Net_results]);
        fclose('all');
        
    end

    mat_filename = ['DOE_R',num2str(n),'.mat']; % Purge out SBCE log file
    save(['MCS_results/',mat_filename], 'MCS_runs', 'x0' )

end
    
