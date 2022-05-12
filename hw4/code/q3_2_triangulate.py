from email.utils import collapse_rfc2231_value
import numpy as np
import matplotlib.pyplot as plt
from tables import Complex128Atom

from helper import camera2
from q2_1_eightpoint import eightpoint
from q3_1_essential_matrix import essentialMatrix

# Insert your package here


'''
Q3.2: Triangulate a set of 2D coordinates in the image to a set of 3D points.
    Input:  C1, the 3x4 camera matrix
            pts1, the Nx2 matrix with the 2D image coordinates per row
            C2, the 3x4 camera matrix
            pts2, the Nx2 matrix with the 2D image coordinates per row
    Output: P, the Nx3 matrix with the corresponding 3D points per row
            err, the reprojection error.

    Hints:
    (1) For every input point, form A using the corresponding points from pts1 & pts2 and C1 & C2
    (2) Solve for the least square solution using np.linalg.svd
    (3) Calculate the reprojection error using the calculated 3D points and C1 & C2 (do not forget to convert from 
        homogeneous coordinates to non-homogeneous ones)
    (4) Keep track of the 3D points and projection error, and continue to next point 
    (5) You do not need to follow the exact procedure above. 
'''
def triangulate(C1, pts1, C2, pts2):
    # Replace pass by your implementation
    pts1_x = pts1[:,0] 
    pts1_y = pts1[:,1]
    pts2_x = pts2[:,0] 
    pts2_y = pts2[:,1]

    # pts1/pts2 should be N x 2

    C11 = C1[0]
    C12 = C1[1]
    C13 = C1[2]

    C21 = C2[0]
    C22 = C2[1]
    C23 = C2[2]

    # P = np.empty(())
    reproj_err = []
    # w_i = []
    P = []
    for i in range(pts1.shape[0]):
        x1,y1 = pts1[i,0],pts1[i,1]
        x2,y2 = pts2[i,0],pts2[i,1]

        A1 = C13*x1 - C11
        A2 = C13*y1 - C12
        A3 = C23*x2 - C21
        A4 = C23*y2 - C22

        A = np.vstack((A1,A2,A3,A4))
        U,S,Vh = np.linalg.svd(A)

        w = Vh[-1]
        w = w/w[-1]
        P.append(w)
        
        # w_i = np.ones((1,pts1.shape[0]))
        # w_i = np.asarray([[P.T],[w_i]])

        new_pts1 = (C1 @ w)
        new_pts2 = (C2 @ w)
        
        points1 = np.insert(pts1[i],2,1)
        points2 = np.insert(pts2[i],2,1)

        new_pts1 = (new_pts1/new_pts1[-1])
        new_pts2 = (new_pts2/new_pts2[-1])

        # points1 = (3,)
        # new_pts1 = (3,)

        # points2 = (3,)
        # new_pts2 = (3,)

        err = np.linalg.norm(points1 - new_pts1) + np.linalg.norm(points2 - new_pts2)
    err = np.sum(err)
    # reproj_err = 
        # reproj_err = np.sum(reproj_err)
    P = np.array(P)
    return P, err

'''
Q3.3:
    1. Load point correspondences
    2. Obtain the correct M2
    3. Save the correct M2, C2, and P to q3_3.npz
'''
def findM2(F, pts1, pts2, K1, K2, filename = 'q3_3.npz'):
    '''
    Q2.2: Function to find the camera2's projective matrix given correspondences
        Input:  F, the pre-computed fundamental matrix
                pts1, the Nx2 matrix with the 2D image coordinates per row
                pts2, the Nx2 matrix with the 2D image coordinates per row
                intrinsics, the intrinsics of the cameras, load from the .npz file
                filename, the filename to store results
        Output: [M2, C2, P] the computed M2 (3x4) camera projective matrix, C2 (3x4) K2 * M2, and the 3D points P (Nx3)
    
    ***
    Hints:
    (1) Loop through the 'M2s' and use triangulate to calculate the 3D points and projection error. Keep track 
        of the projection error through best_error and retain the best one. 
    (2) Remember to take a look at camera2 to see how to correctly reterive the M2 matrix from 'M2s'. 

    '''
    N = pts1.shape[0]
    M2 = np.zeros((3,4))
    C2 = np.zeros((3,4))
    P = np.zeros((N,3))
    M1 = np.hstack((np.eye(3,3),np.zeros((3,1))))
    # K1, K2 = intrinsics['K1'], intrinsics['K2']
    C1 = K1 @ M1
    E = essentialMatrix(F,K1,K2)
    M2_matrices = camera2(E)
    for i in range(4):
        C2 = K2 @ M2_matrices[:,:,i]
        w, err = triangulate(C1, pts1, C2, pts2)

        if min(pts1[:,0]) >= 0 and min(pts2[:,1]) >= 0:
            M2 = M2_matrices[:,:,i]
            P = w
    C2 = K2 @ M2
    return M2, C2, P



if __name__ == "__main__":

    correspondence = np.load('../data/some_corresp.npz') # Loading correspondences
    intrinsics = np.load('../data/intrinsics.npz') # Loading the intrinscis of the camera
    K1, K2 = intrinsics['K1'], intrinsics['K2']
    pts1, pts2 = correspondence['pts1'], correspondence['pts2']
    im1 = plt.imread('../data/im1.png')
    im2 = plt.imread('../data/im2.png')

    F = eightpoint(pts1, pts2, M=np.max([*im1.shape, *im2.shape]))

    M2, C2, P = findM2(F, pts1, pts2, K1,K2)
    np.savez('../results/q3_3.npz',M2,C2,P)

    # Simple Tests to verify your implementation:
    M1 = np.hstack((np.identity(3), np.zeros(3)[:,np.newaxis]))
    C1 = K1.dot(M1)
    C2 = K2.dot(M2)
    P_test, err = triangulate(C1, pts1, C2, pts2)
    print(err)
    assert(err < 500)