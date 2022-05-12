import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from q2_1_eightpoint import eightpoint
from q3_2_triangulate import findM2
from q4_1_epipolar_correspondence import epipolarCorrespondence

# Insert your package here


'''
Q4.2: Finding the 3D position of given points based on epipolar correspondence and triangulation
    Input:  temple_pts1, chosen points from im1
            intrinsics, the intrinsics dictionary for calling epipolarCorrespondence
            F, the fundamental matrix
            im1, the first image
            im2, the second image
    Output: P (Nx3) the recovered 3D points
    
    Hints:
    (1) Use epipolarCorrespondence to find the corresponding point for [x1 y1] (find [x2, y2])
    (2) Now you have a set of corresponding points [x1, y1] and [x2, y2], you can compute the M2
        matrix and use triangulate to find the 3D points. 
    (3) Use the function findM2 to find the 3D points P (do not recalculate fundamental matrices)
    (4) As a reference, our solution's best error is around ~2000 on the 3D points. 
'''
def compute3D_pts(temple_pts1, intrinsics, F, im1, im2):

    # ----- TODO -----
    # YOUR CODE HERE
    # raise NotImplementedError()
    x1 = temple_pts1['x1']
    y1 = temple_pts1['y1']

    x1 = x1[:,0]
    y1 = y1[:,0]
    K1, K2 = intrinsics['K1'], intrinsics['K2']

    x2 = np.zeros(x1.shape[0])
    y2 = np.zeros(x1.shape[0])
    x_2 = []
    y_2 = []
    for i in range(x1.shape[0]):
        x2,y2 = epipolarCorrespondence(im1, im2, F, x1[i], y1[i])
        x_2.append(x2)
        y_2.append(y2)

    
    x_2 = np.asarray(x_2)
    y_2 = np.asarray(y_2)
    pts1 = np.asarray([x1,y1]).T
    pts2 = np.asarray([x_2,y_2]).T
    # pts1 = np.asarray([x1,y1])
    # pts2 = np.asarray([x_2,y_2])

    M2,C2,P = findM2(F,pts1,pts2,K1,K2)
    # P = np.asarray([x2,y2,1])


    return M2,C2,P



'''
Q4.2:
    1. Integrating everything together.
    2. Loads necessary files from ../data/ and visualizes 3D reconstruction using scatter
'''
if __name__ == "__main__":

    temple_coords_path = np.load('../data/templeCoords.npz')
    # print(temple_coords_path)
    correspondence = np.load('../data/some_corresp.npz') # Loading correspondences
    intrinsics = np.load('../data/intrinsics.npz') # Loading the intrinscis of the camera
    K1, K2 = intrinsics['K1'], intrinsics['K2']
    pts1, pts2 = correspondence['pts1'], correspondence['pts2']
    im1 = plt.imread('../data/im1.png')
    im2 = plt.imread('../data/im2.png')


    # ----- TODO -----
    # YOUR CODE HERE
    M1 = np.hstack((np.eye(3,3),np.zeros((3,1))))
    # temple_pts = temple_coords_path
    F = eightpoint(pts1, pts2, M=np.max([*im1.shape, *im2.shape]))
    M2,C2,P = compute3D_pts(temple_coords_path, intrinsics, F, im1, im2)
    C1 = K1 @ M1
    C2 = K2 @ M2
    
    # plt.figure()  
    # plt.scatter(P[:,0],P[:,1],P[:,2])
    fig = plt.figure()



#%%
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    ax.scatter(P[:,0], P[:,1], P[:,2])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    ax.set_xlim(-2,1)
    ax.set_ylim(-1,2)
    # ax.set_zlim(-7,-2)
    ax.set_zlim(-8,-5)

    plt.show()
    # plt.show()
    np.savez('../results/q4_2.npz',F, M1, M2, C1, C2)

# %%
