'''
collection of utility methods shared across files
'''

import os
import numpy as np

def check_folder(folder='render/'):
    '''check if folder exists, make if not present'''
    if not os.path.exists(folder):
            os.makedirs(folder)


def ext_arrs(A,B, precision="float64"):
    nA,dim = A.shape
    A_ext = np.ones((nA,dim*3),dtype=precision)
    A_ext[:,dim:2*dim] = A
    A_ext[:,2*dim:] = A**2

    nB = B.shape[0]
    B_ext = np.ones((dim*3,nB),dtype=precision)
    B_ext[:dim] = (B**2).T
    B_ext[dim:2*dim] = -2.0*B.T
    return A_ext, B_ext

def pairwise_dist(a):
    '''Compute spatial matrix of particles to calculate interactive forces'''
    A_ext, B_ext = ext_arrs(a,a)
    dist = A_ext.dot(B_ext)
    np.fill_diagonal(dist,0)
    return np.sqrt(dist)

def pairwise_comp(A): # Using NumPy broadcasting    
    '''Subtract pairwise coordinates of particles for interactive forces'''
    a = np.asarray(A) # Convert to array if not already so
    mask = a - a[:,None]
    return mask