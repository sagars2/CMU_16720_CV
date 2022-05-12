import numpy as np
import matplotlib.pyplot as plt
from helper import displayEpipolarF, calc_epi_error, toHomogenous, refineF

# Insert your package here



'''
Q2.1: Eight Point Algorithm
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scalar parameter computed as max (imwidth, imheight)
    Output: F, the fundamental matrix

    HINTS:
    (1) Normalize the input pts1 and pts2 using the matrix T.
    (2) Setup the eight point algorithm's equation.
    (3) Solve for the least square solution using SVD. 
    (4) Use the function `_singularize` (provided) to enforce the singularity condition. 
    (5) Use the function `refineF` (provided) to refine the computed fundamental matrix. 
        (Remember to usethe normalized points instead of the original points)
    (6) Unscale the fundamental matrix
'''
def eightpoint(pts1, pts2, M):
    # Replace pass by your implementation
    A_init = np.zeros((pts1.shape[0],9))
    # Normalization step:
    pts1_x = pts1[:,0]/M 
    pts1_y = pts1[:,1]/M
    pts2_x = pts2[:,0]/M 
    pts2_y = pts2[:,1]/M
    #Applying 8-point algorithm
    A = np.asarray([pts1_x*pts2_x, pts1_x*pts2_y, pts1_x,pts1_y*pts2_x,pts1_y*pts2_y,pts1_y,pts2_x,pts2_y,np.ones(pts1.shape[0])]).T
    U,S,Vh = np.linalg.svd(A)
    # print('U',U)
    # print('Vh',Vh)

    F = refineF(np.reshape(Vh[-1],(3,3)), pts1/M, pts2/M)
    T = np.asarray([[1/M, 0, 0],
                    [0, 1/M, 0],
                    [0, 0, 1]])
    F_unnormalized = (T.T @ F @ T)
    return F_unnormalized




if __name__ == "__main__":
        
    correspondence = np.load('../data/some_corresp.npz') # Loading correspondences
    intrinsics = np.load('../data/intrinsics.npz') # Loading the intrinscis of the camera
    K1, K2 = intrinsics['K1'], intrinsics['K2']
    pts1, pts2 = correspondence['pts1'], correspondence['pts2']
    im1 = plt.imread('../data/im1.png')
    im2 = plt.imread('../data/im2.png')

    F = eightpoint(pts1, pts2, M=np.max([*im1.shape, *im2.shape]))
    F = F/F[2,2]

    # Q2.1
    # Write your code here
    displayEpipolarF(im1,im2,F)
    np.savez("../results/q2_1.npz",F)
    # print('F: ',F)
    # Simple Tests to verify your implementation:
    pts1_homogenous, pts2_homogenous = toHomogenous(pts1), toHomogenous(pts2)

    assert(F.shape == (3, 3))
    assert(F[2, 2] == 1)
    assert(np.linalg.matrix_rank(F) == 2)
    assert(np.mean(calc_epi_error(pts1_homogenous, pts2_homogenous, F)) < 1)