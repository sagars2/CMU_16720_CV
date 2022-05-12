# ##################################################################### #
# 16720: Computer Vision Homework 6
# Carnegie Mellon University
# April 12, 2022
# ##################################################################### #

import numpy as np
import matplotlib.pyplot as plt
from q1 import loadData, estimateAlbedosNormals, displayAlbedosNormals, estimateShape, plotSurface 
from q1 import estimateShape
from utils import enforceIntegrability, plotSurface 

def estimatePseudonormalsUncalibrated(I):

    """
    Question 2 (b)

    Estimate pseudonormals without the help of light source directions. 

    Parameters
    ----------
    I : numpy.ndarray
        The 7 x P matrix of loaded images

    Returns
    -------
    B : numpy.ndarray
        The 3 x P matrix of pseudonormals
    
    L : numpy.ndarray
        The 3 x 7 array of lighting directions

    """

    B = None
    L = None
    # Your code here
    U,S,Vt = np.linalg.svd(I,full_matrices=False)
    # Since rank = 3,
    S[3:] = 0
    S = np.diag(S)
    B = Vt[0:3,:]
    L = U[0:3,:]
    return B, L

def plotBasRelief(B, mu, nu, lam):

    """
    Question 2 (f)

    Make a 3D plot of of a bas-relief transformation with the given parameters.

    Parameters
    ----------
    B : numpy.ndarray
        The 3 x P matrix of pseudonormals

    mu : float
        bas-relief parameter

    nu : float
        bas-relief parameter
    
    lambda : float
        bas-relief parameter

    Returns
    -------
        None

    """
    # Your code here
    G = np.array([[1,0,0],[0,1,0],[mu,nu,lam]])
    b_dash = np.linalg.inv(G.T) @ B
    surf = estimateShape(b_dash,s)
    plotSurface(surf)
    # pass

if __name__ == "__main__":
    I, L0, s = loadData('../data/')
    print('L0: ',L0)
    # Part 2 (b)
    # Your code here
    B, L = estimatePseudonormalsUncalibrated(I)
    print('L: ',L)
    #from q1.py
    albedos, normals = estimateAlbedosNormals(B)
    albedoIm, normalIm = displayAlbedosNormals(albedos, normals, s)
    plt.imsave('2b-a.png', albedoIm.clip(0,0.5), cmap = 'gray')
    plt.imsave('2b-b.png', normalIm, cmap = 'rainbow')
    # Part 2 (d)
    # Your code here
    surface = estimateShape(normals,s)
    plotSurface(surface)
    # Part 2 (e)
    # Your code here
    pseudonormals = enforceIntegrability(normals,s)
    surface2 = estimateShape(pseudonormals,s)
    plotSurface(surface2)

    # Part 2 (f)
    # Your code here
    plotBasRelief(pseudonormals, mu = 0.1, nu = 0.1, lam = 0.1)
    plotBasRelief(pseudonormals, mu = 0.1, nu = 0.1, lam = 10)
    plotBasRelief(pseudonormals, mu = 0.1, nu = 10, lam = 0.1)
    plotBasRelief(pseudonormals, mu = 10, nu = 0.1, lam = 0.1)
    plotBasRelief(pseudonormals, mu = 10, nu = 10, lam = 0.1)
    plotBasRelief(pseudonormals, mu = 10, nu = 10, lam = 10)

## Change 0.01 and play around with it 