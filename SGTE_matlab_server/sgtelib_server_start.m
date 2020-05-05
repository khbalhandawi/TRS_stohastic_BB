function sgtelib_server_start(model,keepopen,run_external)

sgtelib_server_stop;

if nargin == 1
    keepopen=false;
    run_external = true;
elseif nargin == 2
    run_external = true;
end

disp('Start sgtelib.exe in server mode.');

termprog = 'bg';

disp(['Selected terminal software: ' termprog]);

% Verbose option of sgtelib.
verboseoption = ''; externaloption = '';
% Option of keep open
if keepopen
    verboseoption = ' -verbose';
end
if run_external
    externaloption = ' &';
end

% command to start sgtelib.
sgtelibcmd = [' sgtelib.exe -server -model ' model verboseoption externaloption];
command = sgtelibcmd;

% Reset ld_library_path
old_ld_library_path = getenv('LD_LIBRARY_PATH');
setenv('LD_LIBRARY_PATH','.');

disp(command)
[status,response] = system(command);
if keepopen
    pause(1);
end
if status || ~isempty(response)
    disp(command);
    disp(status)
    disp(response)
end
pause(5);

% Old LD_LIBRARY_PATH
setenv('LD_LIBRARY_PATH',old_ld_library_path);

