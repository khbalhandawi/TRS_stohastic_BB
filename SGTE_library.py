#==============================================================================#
# %% SERVER START COMMAND

class SGTE_server():
    
    server_process = None
    
    def __init__(self, model):
        self.model = model
    
    def sgtelib_server_stop(self):
        import time
        
        print('Kill sgtelib_server.exe');
        self.system_command('type nul > flag_quit');
        time.sleep(2)
        self.system_command('del flag_quit');
        # !killName sgtelib_server.exe

    def sgtelib_server_start(self,*args): # keepopen, run_external
        
        import time, os
        
        # Remove ALL flag files
        self.system_command('@echo off && for /r %F in (flag_*) do del "%~nF"');
        
        varargin = args
        nargin = len(varargin) # 1 mandatory arguments (model) 
        
        self.sgtelib_server_stop();
    
        if nargin == 0:
            keepopen=False;
            run_external = True;
        elif nargin == 1:
            run_external = True;

        print('Start sgtelib.exe in server mode.');
    
        termprog = 'bg';
    
        print('Selected terminal software: %s' %termprog);
    
        # Verbose option of sgtelib.
        verboseoption = ''; externaloption = '';
        # Option of keep open
        if keepopen:
            verboseoption = ' -verbose';
        if run_external:
            externaloption = ' &';
    
        # command to start sgtelib.
        sgtelibcmd = ' sgtelib.exe -server -model %s %s %s' %(self.model,verboseoption,externaloption);
        command = sgtelibcmd;
    
        # Reset ld_library_path
        try:
            old_ld_library_path = os.environ['LD_LIBRARY_PATH'];
        except KeyError:
            old_ld_library_path = '.'
        os.environ['LD_LIBRARY_PATH'] = '.';
    
        print(command)
        p = self.server_command(command);
        self.server_process = p
        if keepopen:
            # Wait for 5 seconds
            time.sleep(1)
            
        # Wait for 5 seconds
        time.sleep(5)
    
        # Old LD_LIBRARY_PATH
        os.environ['LD_LIBRARY_PATH'] = old_ld_library_path;
    
    def sgtelib_server_newdata(self,X,Z):

        # sgtelib_server_ping;
        
        # Remove all flags
        self.system_command('@echo off && for /r %F in (flag_new_data_*) do del "%~nF"');
        # self.system_command('rm -f flag_new_data_* 2>/dev/null');
        
        # Write matrices
        self.sgtelib_server_write_matrix(X,'X','new_data_x.txt');
        
        self.sgtelib_server_write_matrix(Z,'Z','new_data_z.txt');
        
        # Create flag
        self.system_command('type nul > flag_new_data_transmit');
        
        # Wait for reception flag
        self.sgtelib_server_wait_file('flag_new_data_received');
        
        # Remove all flags
        self.system_command('@echo off && for /r %F in (flag_new_data_*) do del "%~nF"');
        # self.system_command('rm -f flag_new_data_* 2>/dev/null');
        
    def sgtelib_server_write_matrix(self,M,name,file):
            
        fid = open(file,'w');
        fid.write('%s=[\n' %name);
        
        for row in M:
            fid.write('  '.join(['{:12.12f}'.format(x) for x in row]));
            fid.write(';\n');
        
        fid.write('];');
        fid.close()

    def sgtelib_server_wait_file(self,name,*args): # wait_tmax

        import time, os
        varargin = args
        nargin = 1 + len(varargin) # 1 mandatory arguments (model) 
        if not isinstance(name, list):
            name = [name];
        
        if nargin==1:
            wait_tmax = 1000;
        elif nargin==2:
            wait_tmax = varargin[0]
        wait_dt = 0.001;
        wait_t = 0;
        
        while wait_t < wait_tmax:
            time.sleep(wait_dt);
            wait_t = wait_t+wait_dt;
        
            # Check if there is a new input file
            i = 0;
            for item in name:
                i += 1;
                if os.path.exists(item):
                    return i
        
        i = 0;
        if len(name)==1:
            print('sgtelib_server_wait_file: file "%s" not found within time limit' %name[0]);
        else:
            print('sgtelib_server_wait_file: file not found within time limit');

        return i
    
    def sgtelib_server_predict(self,X):
        self.sgtelib_server_ping(1); # wait until surrogate model is built
        
        # Remove flags
        self.system_command('@echo off && for /r %F in (flag_predict_*) do del "%~nF"');
        
        # Write prediction point
        self.sgtelib_server_write_matrix(X,'X','flag_predict_create');
        
        # Create flag
        # self.system_command('copy flag_predict_create debug'); for debugging
        self.system_command('move flag_predict_create flag_predict_transmit');
        
        # Wait for reception flag
        self.sgtelib_server_wait_file('flag_predict_finished');
        self.sgtelib_server_ping(1); # wait until prediction is finished
           
        # Read Output file
        [Z,std,ei,cdf] = self.sgtelib_server_read_matrix('flag_predict_finished',4);
            
        # Remove all flags
        self.system_command('@echo off && for /r %F in (flag_predict_*) do del "%~nF"');
        return Z,std,ei,cdf
    
    def sgtelib_server_cv(self):

        # self.sgtelib_server_ping;
        
        # Remove flags
        self.system_command('@echo off && for /r %F in (flag_cv_*) do del "%~nF"');
        
        # Create flag
        self.system_command('touch flag_cv_transmit');
        
        # Wait for reception flag
        self.sgtelib_server_wait_file('flag_cv_finished');
        
        # Read Output file
        [Zh,Sh,Zv,Sv] = self.sgtelib_server_read_matrix('flag_cv_finished');
        
        # Remove all flags
        self.system_command('@echo off && for /r %F in (flag_cv_*) do del "%~nF"');

        return Zh,Sh,Zv,Sv

    def sgtelib_server_info(self):

        # self.sgtelib_server_ping;
        
        # Remove flags
        self.system_command('@echo off && for /r %F in (flag_info_*) do del "%~nF"');
        
        # Write infoion point
        self.system_command('type nul > flag_info_transmit');
        
        # Wait for reception flag
        self.sgtelib_server_wait_file('flag_info_finished');

    def sgtelib_server_metric(self,metric_str):

        # sgtelib_server_ping;
        
        # Remove flags
        self.system_command('@echo off && for /r %F in (flag_metric_*) do del "%~nF"');
        
        # Write metricion point
        self.system_command('echo %s >> flag_metric_create' %metric_str)
        
        # Create flag
        self.system_command('move flag_metric_create flag_metric_transmit');
         
        # Wait for reception flag
        self.sgtelib_server_wait_file('flag_metric_finished');
         
        # Read Output file
        M = self.sgtelib_server_read_metric('flag_metric_finished');
         
        # Remove all flags
        self.system_command('@echo off && for /r %F in (flag_metric_*) do del "%~nF"');
         
        return M
        
    def sgtelib_server_read_metric(self,file):
        import numpy as np
        fileID = open(file,'r'); # Open file
        InputText = np.loadtxt(fileID,
                           delimiter = ' ',
                           dtype=np.str); # \n is the delimiter
        
        M = InputText[0:-1].astype('float')
        fileID.close()
        
        return M
        
    def sgtelib_server_read_matrix(self,file,nargout):
        import numpy as np
        
        # Get matrices names from the file
        NAMES = [];
        fileID = open(file,'r'); # Open file
        InputText = np.loadtxt(fileID,
                           delimiter = '\n',
                           dtype=np.str); # \n is the delimiter

        # Object separator mark
        StartIndex = np.array([], dtype=int) # Initialize empty header index object
        EndIndex = np.array([], dtype=int) # Initialize empty header index object
        for n,line in enumerate(InputText): # Read line by line
            #Look for object
            if line.find('=') != -1:
                i = line.find('=')
                line = line[0:i];
                line=line.replace(' ','');
                NAMES += [line];
                
                s = np.array(n, dtype=int) # Line number of title
                StartIndex = np.append(StartIndex, [s], axis=0)
                
            elif line.find(']') != -1:
                e = np.array(n, dtype=int) # Line number of title
                EndIndex = np.append(EndIndex, [e], axis=0)
        
        varargout = []
        for H in range(len(StartIndex)): # Blocks to read and tabulate (all selected)
            T = 0; # line counter (inside header block)
            # Read lines in range between two header blocks +1 and -1 from end
            #-------------------------------------------------- >>> n starts here
            for nn in range(int(StartIndex[H]+1), int(EndIndex[H]), 1):
                T += 1
                line = InputText[nn].split()[0:-1]
                line = np.array(line, dtype=float)
                if T ==1:
                    table_out = line # Initialize empty TIME table (block)
                else:
                    table_out = np.vstack((table_out,line))
                    
            varargout += [table_out]
        
        fileID.close()
        N = len(NAMES);
        # Check that the number of output is smaller than the number of matrices in the file
        if N < nargout:
            print('File name: %s' %file);
            print('Nb of matrices in the file: %i' %N);
            print('Nb of matrices required in output: %i' %nargout);
            raise NameError('The file does not contain enough matrices');

        return varargout
    
    def sgtelib_server_ready(self):

        ready = self.sgtelib_server_ping;
        return ready
    
    def sgtelib_server_reset(self):

        # Remove flags
        self.system_command('@echo off && for /r %F in (flag_reset_*) do del "%~nF"');
    
        # Create flag
        self.system_command(['type nul > flag_reset_transmit']);
    
        # Wait for reception flag
        self.sgtelib_server_wait_file('flag_reset_finished');
    
        # Remove all flags
        self.system_command('@echo off && for /r %F in (flag_reset_*) do del "%~nF"');
    
    def sgtelib_server_ping(self,*args): # type
        
        import time
        varargin = args
        nargin = len(varargin) # 1 mandatory arguments (model) 

        if nargin == 0:
            type = 0;
        elif nargin == 1:
            type = varargin[0];
        
        if type == 0:
            wait_time = 2;
        elif type == 1:
            wait_time = 10000;
        
        self.system_command('@echo off && for /r %F in (flag_ping*) do del "%~nF"');
        self.system_command('type nul > flag_ping');
        time.sleep(1)
        # Wait for reception flag
        
        i = self.sgtelib_server_wait_file('flag_pong',wait_time);
        if i==0:
            print('Retry ping...');
            self.system_command('type nul > flag_ping');
            i = self.sgtelib_server_wait_file('flag_pong',wait_time);
        elif i > 0:
            #===================================================================
            # with open('flag_pong') as fp:
            #     ready = fp.readline()
            #===================================================================
            self.system_command('del flag_pong');
            # print('ping ok!');
        else:
            print('=====================SGTELIB_SERVER ERROR==========================');
            print('sgtelib_server not responding');
            raise NameError('We tried to "ping" sgtelib_server, but it is off or not responding');
        
    #==============================================================================#
    # %% Execute system commands and return output to console
    def server_command(self,command):
        import subprocess
        from subprocess import PIPE,STDOUT
        DETACHED_PROCESS = 0x00000008
        CREATE_NO_WINDOW = 0x08000000
        p = subprocess.Popen(command,shell=True, stdin=PIPE, stdout=PIPE, stderr=STDOUT,
                             creationflags=DETACHED_PROCESS) # disable windows errors
        return p
        
    def system_command(self,command):
        import subprocess
        CREATE_NO_WINDOW = 0x08000000
        
        p = subprocess.call(command,shell=True, creationflags = CREATE_NO_WINDOW) # disable windows errors
        
    def server_print(self,p):
        # print server output
        for line in iter(p.stdout.readline, b''):
            line = line.decode('utf-8')
            print(line.rstrip()) # print line by line

if __name__ == "__main__":
    import numpy as np
    
    p = 50; x1max = 5; x2max = 100;
    X = np.concatenate((np.round(np.random.rand(p,1)*x1max), 
                        np.round(np.random.rand(p,1)*x2max)), axis=1);
    X = np.unique(X, axis=0)
    #===========================================================================
    # Z = X[:,1]**2 * X[:,0] * ( 2*np.remainder(X[:,0],2)-1 )
    # Z = Z.reshape((X.shape[0],1))
    #===========================================================================
    
    Z1 = (X[:,1]**1) + (2*X[:,0])
    Z2 = (X[:,1]**1.3) + (2*X[:,0])
    Z = np.concatenate( (Z1.reshape((X.shape[0],1)),
                         Z2.reshape((X.shape[0],1))), axis = 1)
    
    server = SGTE_server('TYPE KS KERNEL_TYPE D1 KERNEL_SHAPE 1.43025 DISTANCE_TYPE NORM1')
    #[Z, std, ei, cdf] = server.sgtelib_server_read_matrix('predict_finished_2',4)
    #server.sgtelib_server_write_matrix(Z,'Z','flag_predict_create');
    # server.sgtelib_server_write_matrix(XX,'X','flag_predict_create');

    p = server.sgtelib_server_start()
    server.sgtelib_server_ping()
    
    server.sgtelib_server_newdata(X,Z)
    
    M = server.sgtelib_server_metric('OECV');
    print(M)
    # Make predictions
    x = np.arange(0,x1max)
    y = np.arange(0,x2max)
    x1, x2 = np.meshgrid(x, y)
    x1 = x1.reshape((x1.size,1)); x2 = x2.reshape((x2.size,1))
    XX = np.concatenate((x1,x2), axis=1);
                 
    [Z, std, ei, cdf] = server.sgtelib_server_predict(XX)
    print(Z)
    server.sgtelib_server_stop()
    server.server_print(server.server_process)
     
    