# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 01:34:24 2017

@author: Khalil
"""
import sys, os, pickle, time
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from matplotlib import rc
from visualization import gridsamp, hyperplane_SGTE_vis_norm

#==============================================================================#
# SCALING BY A RANGE
def scaling(x,l,u,operation):
    ''' 
    scaling() scales or unscales the vector x according to the bounds
    specified by u and l. The flag type indicates whether to scale (1) or
    unscale (2) x. Vectors must all have the same dimension.
    '''
    
    if operation == 1:
        # scale
        x_out=(x-l)/(u-l)
    elif operation == 2:
        # unscale
        x_out = l + x*(u-l)
    
    return x_out

#==============================================================================#
# POSTPROCESS DOE DATA
def train_server(training_X, training_Y, bounds):
    
    from visualization import define_SGTE_model
    from SGTE_library import SGTE_server
   
    #======================== SURROGATE META MODEL ============================#
    # %% SURROGATE modeling
    lob = bounds[:,0]
    upb = bounds[:,1]
    
    Y = training_Y; S_n = scaling(training_X, lob, upb, 1)
    # fitting_names = ['KRIGING','LOWESS','KS','RBF','PRS','ENSEMBLE']
    # run_types = ['optimize hyperparameters','load hyperparameters']
    fit_type = 1; run_type = 1 # optimize all hyperparameters
    model,sgt_file = define_SGTE_model(fit_type,run_type)
    server = SGTE_server(model)
    server.sgtelib_server_start()
    server.sgtelib_server_ping()
    server.sgtelib_server_newdata(S_n,Y)    
    #===========================================================================
    # M = server.sgtelib_server_metric('RMSECV')
    # print('RMSECV Metric: %f' %(M[0]))
    #===========================================================================    

    return server

def visualize_surrogate(bounds,variable_lbls,server,training_X,
                        training_Y,plt,current_path=os.getcwd(),vis_output=0,
                        resolution=20,output_lbls="$\hat{f}(\mathbf{x})$",
                        base_name="surrogate_model"):

    rc('text', usetex=True)
    #===========================================================================
    # Plot 2D projections
    nominal = [0.5]*len(bounds[:,0]); nn = resolution
    fig = plt.figure()  # create a figure object

    hyperplane_SGTE_vis_norm(server,training_X,bounds,variable_lbls,nominal,training_Y,nn,fig,plt)
    
    fig_name = '%s.pdf' %(base_name)
    fig_file_name = os.path.join(current_path,fig_name)
    fig.savefig(fig_file_name, bbox_inches='tight')

#------------------------------------------------------------------------------#
# MAIN FILE
def main():

    #============================== TRAINING DATA =================================#
    # one-liner to read a single variable
    mat = loadmat('DOE_V1.mat') # get matlab data

    # Variables
    ax_pos = mat['ax_pos']
    st_height = mat['st_height']
    st_width = mat['st_width']
    laser_power = mat['laser_power']

    # Parameters
    T1_n = mat['T1_n']
    T2_n = mat['T2_n']
    T3_n = mat['T3_n']
    T4_n = mat['T4_n']
    shroud_width = mat['shroud_width']

    # Combined design/parameter space
    training_X = np.column_stack((ax_pos, st_height, st_width, laser_power,
                                  T1_n, T2_n, T3_n, T4_n, shroud_width))

    # Outputs
    n_f_th = mat['n_f_th']
    temp = mat['temp']
    
    training_Y = np.column_stack((n_f_th, temp))

    #============================= MAIN EXECUTION =================================#
    
    # plt.rc('text', usetex=True)
    start_time = time.time()

    
    bounds = np.array( [[45.0   , 155.0],   # Axial Position #1
                        [2.0    , 20   ],   # Stiff height   #2
                        [20.0   , 155.0],   # Stiff width    #3
                        [3500   , 4500 ],   # Laser power    #4
                        [-100.0 , 100.0],   # T1             #5
                        [-100.0 , 100.0],   # T2             #6
                        [-100.0 , 100.0],   # T3             #7
                        [-100.0 , 100.0],   # T4             #8
                        [190.0  , 250.0]] ) # Shroud width   #9

    # Define design space bounds
    bounds_v = bounds[:4,:]
    print(bounds_v)

    # Define parameter space bounds
    bounds_p = bounds[4::,:]
    print(bounds_p)

    variable_lbls = ['Axial $x_1$ (mm)','Height $x_2$ (mm)','Width $x_3$ (mm)','Power $P_L$ (W)']
    output_lbls = ['Safety factor - $n_f$', 'Temperature - $T$ ($^o$C)']

    # Train surrogate model for use in subsequent plotting
    server = train_server(training_X,training_Y,bounds)
    #===========================================================================
    # Visualize surrogate model
    resolution = 20
    vis_output = 0

    visualize_surrogate(bounds,variable_lbls,server,training_X,
                        training_Y,plt,resolution=resolution)

    plt.show()
        
if __name__ == '__main__':
    main()
