function M = sgtelib_server_info

% sgtelib_server_ping;

% Remove flags
% system('rm -f flag_info_* 2>/dev/null');
system('@echo off && for /r %F in (flag_info_*) do rm -f "%~nF"');

% Write infoion point
system('touch flag_info_transmit');

% Wait for reception flag
sgtelib_server_wait_file('flag_info_finished');
system('@echo off && for /r %F in (flag_info_*) do rm -f "%~nF"');
