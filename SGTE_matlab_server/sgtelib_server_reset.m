function sgtelib_server_reset

% sgtelib_server_ping;

% Remove flags
% system('rm -f flag_reset_* 2>/dev/null');
system('@echo off && for /r %F in (flag_reset_*) do del "%~nF"');

% Create flag
system(['type nul > flag_reset_transmit']);

% Wait for reception flag
sgtelib_server_wait_file('flag_reset_finished');

% Remove all flags
% system('rm -f flag_reset_* 2>/dev/null');
system('@echo off && for /r %F in (flag_reset_*) do del "%~nF"');

