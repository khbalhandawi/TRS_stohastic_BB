function sgtelib_server_stop
disp('Kill sgtelib_server.exe');
system('type nul > flag_quit');
pause(2)
system('del flag_quit');
%!killName sgtelib_server.exe
