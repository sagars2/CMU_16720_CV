from webbrowser import MacOSX
import numpy as np
import matplotlib.pyplot as plt

from helper import displayEpipolarF, calc_epi_error, toHomogenous
from q2_1_eightpoint import eightpoint
from q2_2_sevenpoint import sevenpoint
from q3_2_triangulate import findM2

import scipy

# Insert your package here


# Helper functions for this assignment. DO NOT MODIFY!!!
"""
Helper functions.

Written by Chen Kong, 2018.
Modified by Zhengyi (Zen) Luo, 2021
"""
def plot_3D_dual(P_before, P_after):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title("Blue: before; red: after")
    ax.scatter(P_before[:,0], P_before[:,1], P_before[:,2], c = 'blue')
    ax.scatter(P_after[:,0], P_after[:,1], P_after[:,2], c='red')
    while True:
        x, y = plt.ginput(1, mouse_stop=2)[0]
        plt.draw()


'''
Q5.1: RANSAC method.
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scaler parameter
            nIters, Number of iterations of the Ransac
            tol, tolerence for inliers
    Output: F, the fundamental matrix
            inliers, Nx1 bool vector set to true for inliers

    Hints:
    (1) You can use the calc_epi_error from q1 with threshold to calcualte inliers. Tune the threshold based on 
        the results/expected number of inliners. You can also define your own metric. 
    (2) Use the seven point alogrithm to estimate the fundamental matrix as done in q1
    (3) Choose the resulting F that has the most number of inliers
    (4) You can increase the nIters to bigger/smaller values
 
'''
def ransacF(pts1, pts2, M, nIters=100, tol=1):
    # Replace pass by your implementation
    N = pts1.shape[0]
    max_inliers = -1
    F_array = []
    temp_inliers = None
    for i in range(nIters):
        idx = np.random.choice(pts1.shape[0],8,False)
        F = eightpoint(pts1[idx],pts2[idx],M)
        F_array.append(F)
        
        pts1_homogenous, pts2_homogenous = toHomogenous(pts1), toHomogenous(pts2)
        error = calc_epi_error(pts1_homogenous,pts2_homogenous,F)
        temp_inliers = error<tol
        if temp_inliers[temp_inliers].shape[0] > max_inliers:
            max_inliers = temp_inliers[temp_inliers].shape[0]
            F = F
            inliers = temp_inliers
    F = F/F[2,2]
    return F, inliers

'''
Q5.2: Rodrigues formula.
    Input:  r, a 3x1 vector
    Output: R, a rotation matrix
'''
def rodrigues(r):
    # Replace pass by your implementation
    theta = np.linalg.norm(r)
    I = np.eye(3)
    if theta == 0:
        R = I
    else:
        u = r/theta
        u = u.reshape(3,1)
        u_x = np.asarray([[0,-u[2], u[1]],[u[2], 0, -u[0]],[-u[1], u[0], 0]],dtype= object)
        R = I*np.cos(theta) + (1-np.cos(theta))* (u@u.T) + u_x*np.sin(theta)
    return R



'''
Q5.2: Inverse Rodrigues formula.
    Input:  R, a rotation matrix
    Output: r, a 3x1 vector
'''
def invRodrigues(R):
    # Replace pass by your implementation
    A = (R-R.T)/2
    I = np.eye(3)
    rho = np.asarray([A[2,1],A[0,2],A[1,0]]).T
    s = np.linalg.norm(rho)
    c = (R[0,0]+R[1,1]+R[2,2]-1)/2
    theta = np.arctan2(s,c)
    if s == 0 and c == 0:
        r = np.zeros((1,3))
    elif s == 0 and c == -1:
        v = R+I
        if np.linalg.norm(v[:,0]) != 0:
            v = v[:,0]
        elif np.linalg.norm(v[:,1]) != 0:
            v = v[:,1]
        else:
            v = v[:,2]
        u = v/np.linalg.norm(v,2)
        r = u*np.pi
        if np.linalg.norm(r) == np.pi and ((r[0,0] == 0 and r[1,0] == 0 and r[2,0]< 0) or (r[0,0] == 0 and r[1,0] < 0) or r[0,0]< 0):
            r = -r
        else:
            r = r
    elif np.sin(theta) != 0:
        u = rho/s
        r = u*theta
    return r


'''
Q5.3: Rodrigues residual.
    Input:  K1, the intrinsics of camera 1
            M1, the extrinsics of camera 1
            p1, the 2D coordinates of points in image 1
            K2, the intrinsics of camera 2
            p2, the 2D coordinates of points in image 2
            x, the flattened concatenationg of P, r2, and t2.
    Output: residuals, 4N x 1 vector, the difference between original and estimated projections
'''
def rodriguesResidual(K1, M1, p1, K2, p2, x):
    # Replace pass by your implementation
    t = x[-3,:].reshape((3,1))
    r = x[-6:-3].reshape((3,1))
    P = x[:,-6].reshape(3,1)
    R = rodrigues(r)
    C1 = K1 @ M1
    M2 = np.hstack((R,t))
    C2 = K2 @ M2
    p1_hat1 = C1 @ P
    p1_hat = p1_hat1/p1_hat1[2,:]
    p2_hat2 = C2 @ P
    p2_hat = p2_hat2/p2_hat2[2,:]
    residuals = np.concatenate([(p1-p1_hat).reshape([-1]), (p2-p2_hat).reshape([-1])])
    return residuals


'''
Q5.3 Bundle adjustment.
    Input:  K1, the intrinsics of camera 1
            M1, the extrinsics of camera 1
            p1, the 2D coordinates of points in image 1
            K2,  the intrinsics of camera 2
            M2_init, the initial extrinsics of camera 1
            p2, the 2D coordinates of points in image 2
            P_init, the initial 3D coordinates of points
    Output: M2, the optimized extrinsics of camera 1
            P2, the optimized 3D coordinates of points
            o1, the starting objective function value with the initial input
            o2, the ending objective function value after bundle adjustment

    Hints:
    (1) Use the scipy.optimize.minimize function to minimize the objective function, rodriguesResidual. 
        You can try different (method='..') in scipy.optimize.minimize for best results. 
'''
def bundleAdjustment(K1, M1, p1, K2, M2_init, p2, P_init):
    # Replace pass by your implementation

    obj_start = obj_end = 0
    # ----- TODO -----
    # YOUR CODE HERE
    # raise NotImplementedError()
    R = M2_init[:,:3]
    t = M2_init[:,3]
    r = invRodrigues(R)


    vec = np.hstack((P_init.flatten(),r.flatten(),t.flatten()))
    func = lambda x: rodriguesResidual(K1, M1, p1, K2, p2, x)

    #find optimized vector 
    vec_optim,_ = scipy.optimize.least_squares(func, vec)

    P = vec_optim[0:-6]
    r1 = vec_optim[-6:-3]
    t1 = vec_optim[-3:]

    R1 = rodrigues(r1)
    M2 = np.hstack((R,t1.reshape(3,1)))
    return M2, P, obj_start, obj_end



if __name__ == "__main__":
              
    np.random.seed(1) #Added for testing, can be commented out

    some_corresp_noisy = np.load('../data/some_corresp_noisy.npz') # Loading correspondences
    intrinsics = np.load('../data/intrinsics.npz') # Loading the intrinscis of the camera
    K1, K2 = intrinsics['K1'], intrinsics['K2']
    noisy_pts1, noisy_pts2 = some_corresp_noisy['pts1'], some_corresp_noisy['pts2']
    im1 = plt.imread('../data/im1.png')
    im2 = plt.imread('../data/im2.png')

    F, inliers = ransacF(noisy_pts1, noisy_pts2, M=np.max([*im1.shape, *im2.shape]))

    # YOUR CODE HERE


    # Simple Tests to verify your implementation:
    pts1_homogenous, pts2_homogenous = toHomogenous(noisy_pts1), toHomogenous(noisy_pts2)

    assert(F.shape == (3, 3))
    assert(F[2, 2] == 1)
    assert(np.linalg.matrix_rank(F) == 2)
    

    # YOUR CODE HERE
    M1 = np.hstack(((np.eye(3)), np.zeros((3, 1))))
    
    M2_init,C2,P_init = findM2(F, noisy_pts1[inliers], noisy_pts2[inliers], K1, K2)
    
    
    # p1 = 
    # M2_init = 
    # p2 = 
    # P_init = 
    M2, P_after, obj_start, obj_end = bundleAdjustment(K1, M1, noisy_pts1[inliers], K2, M2_init, noisy_pts2[inliers], P_init)
    plot_3D_dual(P_init,P_after)
    
    # Simple Tests to verify your implementation:
    from scipy.spatial.transform import Rotation as sRot
    rotVec = sRot.random()
    mat = rodrigues(rotVec.as_rotvec())

    assert(np.linalg.norm(rotVec.as_rotvec() - invRodrigues(mat)) < 1e-3)
    assert(np.linalg.norm(rotVec.as_matrix() - mat) < 1e-3)



    # YOUR CODE HERE