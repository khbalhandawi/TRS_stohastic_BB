function ready = sgtelib_server_ping(type)

if nargin==0
    type = 0; 
end 

if type == 0
    wait_time = 2;
elseif type == 1
    wait_time = 1e11;
end

system('@echo off && for /r %F in (flag_ping*) do del "%~nF"');
system('type nul > flag_ping');
pause(1)
% Wait for reception flag

i = sgtelib_server_wait_file('flag_pong',wait_time);
if i==0
    disp('Retry ping...');
    system('type nul > flag_ping');
    i = sgtelib_server_wait_file('flag_pong',wait_time);
end

if i
    ready = importdata('flag_pong');
    system('del flag_pong');
    %disp('ping ok!');
else
    disp('=====================SGTELIB_SERVER ERROR==========================');
    disp('sgtelib_server not responding');
    error('We tried to "ping" sgtelib_server, but it is off or not responding');
end


