function [Z,std,ei,cdf] = sgtelib_server_predict(X)

sgtelib_server_ping(1); % wait until surrogate model is built

% Remove flags
system('@echo off && for /r %F in (flag_predict_*) do del "%~nF"');

% Write prediction point
sgtelib_server_write_matrix(X,'X','flag_predict_create');

% Create flag
system('move flag_predict_create flag_predict_transmit >nul');

% Wait for reception flag
sgtelib_server_wait_file('flag_predict_finished');

sgtelib_server_ping(1); % wait until prediction is finished

% Read Output file
[Z,std,ei,cdf] = sgtelib_server_read_matrix('flag_predict_finished');

% Remove all flags
system('@echo off && for /r %F in (flag_predict_*) do del "%~nF"');