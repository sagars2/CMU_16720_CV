from locale import ABMON_10
import numpy as np
import matplotlib.pyplot as plt

from helper import displayEpipolarF, calc_epi_error, toHomogenous, _singularize, refineF

# Insert your package here
import sympy as sym

'''
Q2.2: Seven Point Algorithm for calculating the fundamental matrix
    Input:  pts1, 7x2 Matrix containing the corresponding points from image1
            pts2, 7x2 Matrix containing the corresponding points from image2
            M, a scalar parameter computed as max (imwidth, imheight)
    Output: Farray, a list of estimated 3x3 fundamental matrixes.
    
    HINTS:
    (1) Normalize the input pts1 and pts2 scale paramter M.
    (2) Setup the seven point algorithm's equation.
    (3) Solve for the least square solution using SVD. 
    (4) Pick the last two colum vector of vT.T (the two null space solution f1 and f2)
    (5) Use the singularity constraint to solve for the cubic polynomial equation of  F = a*f1 + (1-a)*f2 that leads to 
        det(F) = 0. Sovling this polynomial will give you one or three real solutions of the fundamental matrix. 
        Use np.polynomial.polynomial.polyroots to solve for the roots
    (6) Unscale the fundamental matrixes and return as Farray
'''
def sevenpoint(pts1, pts2, M):

    Farray = []
    # ----- TODO -----
    # YOUR CODE HERE
    pts1 = pts1/M
    pts2 = pts2/M

    pts1_x = pts1[:,0] 
    pts1_y = pts1[:,1]
    pts2_x = pts2[:,0] 
    pts2_y = pts2[:,1]

    A = np.asarray([pts1_x*pts2_x,pts1_x*pts2_y,pts1_x,pts1_y*pts2_x,pts1_y*pts2_y,pts1_y,pts2_x,pts2_y,np.ones(pts1.shape[0])]).T
    U,S,Vh = np.linalg.svd(A)
    
    #Computing the null spaces 
    F1 = np.reshape(Vh[-1,:],(3,3))
    F2 = np.reshape(Vh[-2,:],(3,3))

    #Computing the cubic polynomial
    # a = sym.Symbol('a')
    fun = lambda alpha: np.linalg.det((alpha*F1)+(1-alpha)*F2)
    a0 = fun(0)
    a1 = (2/3)*(fun(1)-fun(-1))-((1/12)*(fun(2)-fun(-2)))
    a2 = (1/2)*(fun(1)+fun(-1)) - a0
    a3 = (fun(1)-fun(-1))*(1/2) - a1
    a_root = np.polynomial.polynomial.polyroots((a0,a1,a2,a3))
    # print(a_root)
    real_root = np.isreal(a_root)
    a_root = np.real(a_root[real_root])     
    T = np.asarray([[1/M, 0, 0],[0, 1/M, 0],[0, 0, 1]])
    Farray = []
    for a in a_root:
        F = a*F1 +(1-a)*F2
        # F = refineF(F,pts1,pts2)
        F_unscaled = (T.T @ F @ T).T
        Farray.append(F_unscaled/F_unscaled[2,2])
    # raise NotImplementedError()
    return Farray
    
    #sum of Farray = 0.14339
    #M = 640
    #roots = 2.5598 1.1649 0.1399
    #coeffs 0.1072, -0.4144, 0.3757, -0.0447


if __name__ == "__main__":
        
    correspondence = np.load('../data/some_corresp.npz') # Loading correspondences
    intrinsics = np.load('../data/intrinsics.npz') # Loading the intrinscis of the camera
    K1, K2 = intrinsics['K1'], intrinsics['K2']
    pts1, pts2 = correspondence['pts1'], correspondence['pts2']
    im1 = plt.imread('../data/im1.png')
    im2 = plt.imread('../data/im2.png')

    # ----- TODO -----
    # YOUR CODE HERE

    # pts1 = np.asarray([[157,231],[158,137],[202,211],[447,393],[200,153],[223,310],[421,236]])
    # pts2 = np.asarray([[157,211],[201,190],[160,238],[234,347],[161,142],[237,256],[55,175]])

    
    # Simple Tests to verify your implementation:
    # Test out the seven-point algorithm by randomly sampling 7 points and finding the best solution. 
    np.random.seed(1) #Added for testing, can be commented out

    pts1_homogenous, pts2_homogenous = toHomogenous(pts1), toHomogenous(pts2)

    max_iter = 500
    pts1_homo = np.hstack((pts1, np.ones((pts1.shape[0], 1))))
    pts2_homo = np.hstack((pts2, np.ones((pts2.shape[0], 1))))

    ress = []
    F_res = []
    choices = []
    M=np.max([*im1.shape, *im2.shape])
    for i in range(max_iter):
        choice = np.random.choice(range(pts1.shape[0]), 7)
        pts1_choice = pts1[choice, :]
        pts2_choice = pts2[choice, :]
        Fs = sevenpoint(pts1_choice, pts2_choice, M)
        for F in Fs:
            choices.append(choice)
            res = calc_epi_error(pts1_homo,pts2_homo, F)
            F_res.append(F)
            ress.append(np.mean(res))
            
    min_idx = np.argmin(np.abs(np.array(ress)))
    F = F_res[min_idx]
    displayEpipolarF(im1,im2,F)
    print("Error:", ress[min_idx])
    print('F',F)
    np.savez('../results/q2_2.npz',F)
    assert(F.shape == (3, 3))
    assert(F[2, 2] == 1)
    assert(np.linalg.matrix_rank(F) == 2)
    assert(np.mean(calc_epi_error(pts1_homogenous, pts2_homogenous, F)) < 1)