function [f] = Blackbox_call(d,sur,P)
    global index index_total
    if (nargin==2)
        sur=false; % default use the real model
    end
    d = d'; % WHEN USING NOMAD ONLY
    %% Simulation Paramters
    input_filename_pool = 'DED_vars_inputs_pool.log';
    param_filename_pool = 'DED_parameters_pool.log';
    input_filename = 'DED_vars_inputs.log';
    param_filename = 'DED_parameters.log';
    err_filename = 'err_out_Blackbox.log';
    out_vars_filename = 'var_out_Blackbox.log';
    sur_filename = 'sur_results.log';
    
%     P = {mesh_size,mesh_size_AM,bais_size,lob_v,upb_v,lob_p,upb_p,...
%          fig1,fig2,fig3,fig4,fig5,T_melt,H_subs,T_ref,b_thick,scanning_speed,...
%          power_density,SBCE_parameters};

    mesh_size = P{1}; mesh_size_AM = P{2}; lob_v = P{3}; 
    upb_v = P{4}; lob_p = P{5}; upb_p = P{6}; 
    fig1 = P{7}; fig5 = P{8};
    T_melt = P{9}; H_subs = P{10}; T_ref = P{11}; b_thick = P{12};
    scanning_speed = P{13}; power_density = P{14}; Total_width = P{15};
    P_SBCE = P{16};

    %% Scale variables
    d = scaling(d, lob_v, upb_v, 2); % Normalize variables for optimization
    P_SBCE = scaling(P_SBCE, lob_p, upb_p, 2); % Normalize parameters for optimization
    
    lob = [lob_v lob_p]; upb = [upb_v upb_p];
    d_SURR = [d P_SBCE]; % variable vector for surrogate evaluation
    
    % Report Model Inputs
    % fprintf("\n#======================================================#\n");
    % fprintf(  "#                         I%2d                          #\n",index);
    % fprintf(  "#======================================================#\n");
    % fprintf("VARIABLES:\n");
    % fprintf("AX: %.4f\n",d_SURR(1));
    % fprintf("TH: %.4f\n",d_SURR(2));
    % fprintf("WI: %.4f\n",d_SURR(3));
    % fprintf("LP: %.4f\n",d_SURR(4));
    % fprintf("PARAMETERS:\n");
    % fprintf("T1: %.4f\n",d_SURR(5));
    % fprintf("T2: %.4f\n",d_SURR(6));
    % fprintf("T3: %.4f\n",d_SURR(7));
    % fprintf("T4: %.4f\n",d_SURR(8));
    % fprintf("IP: %.4f\n",d_SURR(9));
    % fprintf(  "#======================================================#\n");
    
    if ~(sur)
        %% Run pool model
        d_pool = [d(4) scanning_speed power_density];
        [pool_dim, max_T_k] = call_pool_model( d_pool,index,T_melt,H_subs,T_ref,input_filename_pool,param_filename_pool );

        pool_l = pool_dim(:,1); % Length
        pool_d = pool_dim(:,2); % Depth
        pool_w = pool_dim(:,3); % Width
        max_T_k = max_T_k; % peak temperature

        n_layers = round(d(2)/pool_d);
        n_deposit = ceil(d(3)/pool_w);

        fprintf('Number of layers: %f\n',n_layers)
        fprintf('Number of deposits: %f\n',n_deposit)
    
        %% Input Files
        % recombine parameters with design variables
        d_BB = [d, scanning_speed, power_density, P_SBCE, pool_l, pool_w,...
             pool_d, max_T_k, n_layers, n_deposit]; 

        in_full_filename = ['./Optimization_studies/',input_filename];
        if exist(in_full_filename, 'file') == 2
          delete(in_full_filename)
        end
        fileID_var = fopen(in_full_filename,'w');

        Net_results = sprintf('%.4f ' , d_BB);
        Net_results = Net_results(1:end-1);% strip final comma
        fprintf(fileID_var, '%s\n', Net_results);

        fclose(fileID_var);
        fclose('all');

        %% Prepare parameter files
        fileID_param = fopen(['./Optimization_studies/',param_filename],'w');

        Net_results = sprintf('%0.2f ' , [mesh_size,mesh_size_AM,T_melt,H_subs,T_ref,b_thick]);
        Net_results = Net_results(1:end-1);% strip final comma
        fprintf(fileID_param, '%i %s\n', [index, Net_results]);

        fclose(fileID_param);
        fclose('all');
    end
    %% Delete output files
    out_full_filename = ['Optimization_studies/',out_vars_filename];
    if exist(out_full_filename, 'file') == 2
      delete(out_full_filename)
    end

    %% Run Blackbox
    index_total = index_total + 1;
    cstr = d(1) + d(3) - Total_width;
    %%%%%%%%%%%%%%%%%%%%%
    if ~(sur)
        %%%%%%%%%%%%%%%%%%%%%
        % Real model
        %%%%%%%%%%%%%%%%%%%%%
        command = ['python DED_Blackbox_opt.py -- ',input_filename, ' ', param_filename];
        fprintf([command,'\n'])
        status = system(command);
        if exist(out_full_filename, 'file') == 2
            out_exist = 1;
        else
            out_exist = 0;
        end
        %%%%%%%%%%%%%%%%%%%%%
    else
        %%%%%%%%%%%%%%%%%%%%%
        % SAO only
        %%%%%%%%%%%%%%%%%%%%%
        out_exist = 1;
        status = 0;
        %Prediction
        [Z,~,~,~] = sgtelib_server_predict(scaling(d_SURR, lob, upb, 1));
        f1 = Z(1); f5 = Z(2);
        f = [-f1, ...
             T_melt - f5, ...
             cstr];
        %%%%%%%%%%%%%%%%%%%%%
    end
    %%%%%%%%%%%%%%%%%%%%%
    % define point colors
    red = [1 0 0];
    magenta = [178, 102, 255]/256;
    
    if status == 0 & out_exist == 1 & cstr <= 0 % REMOVE CSTR FOR BB OPT
        %% Obtain output
        index = index+1;
        if ~(sur)
            %%%%%%%%%%%%%%%%%%%%%
            % Real model Only
            %%%%%%%%%%%%%%%%%%%%%
            fileID_out = fopen(out_full_filename,'r');
            f = textscan(fileID_out,'%f %f %f', 'Delimiter', ' ');
            f = cell2mat(f);
            fclose(fileID_out);
            fclose('all');
            %%%%%%%%%%%%%%%%%%%%%
        end
        %% Display optimization progress
        if ~(sur) % Color code for bb  points
            append_point(d,'x',magenta,8,1.5,fig1);
            append_point(d,'x',magenta,8,1.5,fig5);
        else % Color code for surrogate points
            append_point(d,'x',red,8,1.5,fig1);
            append_point(d,'x',red,8,1.5,fig5);
            
            fileID_sur = fopen(['Optimization_studies/',sur_filename],'at');
            Net_results = sprintf('%f,' , d);
            obj_results = sprintf('%f,' , f);
            obj_results = obj_results(1:end-1);% strip final comma
            fprintf(fileID_sur, '%i,%s,%s', [index,Net_results,obj_results]);
            fprintf(fileID_sur,'\n');
            fclose('all');
        end
    elseif status == 1 | out_exist == 0 | cstr > 0 % REMOVE CSTR FOR BB OPT
        %% Error execution
        
        fileID_err = fopen(['Optimization_studies/',err_filename],'at');
        Net_results = sprintf('%f,' , d);
        Net_results = Net_results(1:end-1);% strip final comma
        fprintf(fileID_err, '%i,%s', [index,Net_results]);
        fprintf(fileID_err,'\n');
        fclose('all');
        % Color code for invalid points
        if ~(sur) % Color code for bb  points
            append_point(d,'x',magenta,8,1.5,fig1);
            append_point(d,'x',magenta,8,1.5,fig5);
        else % Color code for surrogate points
            append_point(d,'x',red,8,1.5,fig1);
            append_point(d,'x',red,8,1.5,fig5);
        end
        msg = 'Error: Invalid point';
        f = [NaN, NaN, cstr];
        % warning(msg)
    end
    %% Capture progress
end