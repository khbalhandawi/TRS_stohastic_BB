function [f] = TRS_BB_simple(x)
    
    %   Scaling the function inputs from unity 
    
    % Model variables
    bounds = [45.0           , 155.0                    ;... % Axial Position
              2.0            , 20                       ;... % Stiff height
              20.0           , 155.0                    ;... % Stiff width
              -100.0         , 100.0                    ;... % T1
              -100.0         , 100.0                    ;... % T2
              -100.0         , 100.0                    ;... % T3
              -100.0         , 100.0                    ];   % T4

    lob_v = bounds(1:3,1)'; upb_v = bounds(1:3,2)';
    lob_p = bounds(4:end,1)'; upb_p = bounds(4:end,2)';
    
    Total_width = 201.0;
    T_melt = 1500;
    
    lob = [lob_v lob_p]'; upb = [upb_v upb_p]';  
    
    %% Scale variables
    x = scaling(x, lob_v, upb_v, 2); % Normalize variables for optimization
    
    P_random = rand(size(lob_p));
    P_random = scaling(P_random, lob_p, upb_p, 2); % Normalize parameters for optimization
    
    d_SURR = [x P_random]';
    % variable vector for surrogate evaluation
    
    %%%%%%%%%%%%%%%%%%%%%
    % SAO only
    %%%%%%%%%%%%%%%%%%%%%
    %Prediction
    [Z,~,~,~] = sgtelib_server_predict(scaling(d_SURR, lob, upb, 1)');
    
    % cstr = x(1) + x(3) - Total_width;
    % f1 = Z(1); f5 = Z(2);
    % f = [-f1, ...
    %      T_melt - f5, ...
    %      cstr];
     
    cstr = x(1) + x(3) - Total_width;
    f1 = Z(1);
    f = [-f1, ...
         cstr];

end