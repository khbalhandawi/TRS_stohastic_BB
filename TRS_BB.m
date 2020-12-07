function [f] = TRS_BB(x,sur,p)
    
    if (nargin==2)
        p = sur;
        sur=false;
    end
    
    global index
    index = p{1};
    index = index + 1;
    % fprintf('index : %i\n',index)
    
    %   Scaling the function inputs from unity 
    lob_v = p{2};
    upb_v = p{3};
    lob_p = p{4};
    upb_p = p{5};
    T_melt = p{6};
    
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
     
    cstr = x(1) + x(3) - P_random(5);
    f1 = Z(1);
    f = [-f1, ...
         cstr];

end