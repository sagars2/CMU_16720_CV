from tkinter import W
import numpy as np
import matplotlib.pyplot as plt

from helper import _epipoles, displayEpipolarF

from q2_1_eightpoint import eightpoint

# Insert your package here


# Helper functions for this assignment. DO NOT MODIFY!!!
def epipolarMatchGUI(I1, I2, F):
    e1, e2 = _epipoles(F)

    sy, sx, _ = I2.shape

    f, [ax1, ax2] = plt.subplots(1, 2, figsize=(12, 9))
    ax1.imshow(I1)
    ax1.set_title('Select a point in this image')
    ax1.set_axis_off()
    ax2.imshow(I2)
    ax2.set_title('Verify that the corresponding point \n is on the epipolar line in this image')
    ax2.set_axis_off()

    while True:
        plt.sca(ax1)
        # x, y = plt.ginput(1, mouse_stop=2)[0]

        out = plt.ginput(1, timeout=3600, mouse_stop=2)

        if len(out) == 0:
            print(f"Closing GUI")
            break
        
        x, y = out[0]

        xc = int(x)
        yc = int(y)
        v = np.array([xc, yc, 1])
        l = F.dot(v)
        s = np.sqrt(l[0]**2+l[1]**2)

        if s==0:
            print('Zero line vector in displayEpipolar')

        l = l/s

        if l[0] != 0:
            ye = sy-1
            ys = 0
            xe = -(l[1] * ye + l[2])/l[0]
            xs = -(l[1] * ys + l[2])/l[0]
        else:
            xe = sx-1
            xs = 0
            ye = -(l[0] * xe + l[2])/l[1]
            ys = -(l[0] * xs + l[2])/l[1]

        # plt.plot(x,y, '*', 'MarkerSize', 6, 'LineWidth', 2);
        ax1.plot(x, y, '*', markersize=6, linewidth=2)
        ax2.plot([xs, xe], [ys, ye], linewidth=2)

        # draw points
        x2, y2 = epipolarCorrespondence(I1, I2, F, xc, yc)
        ax2.plot(x2, y2, 'ro', markersize=8, linewidth=2)
        plt.draw()


'''
Q4.1: 3D visualization of the temple images.
    Input:  im1, the first image
            im2, the second image
            F, the fundamental matrix
            x1, x-coordinates of a pixel on im1
            y1, y-coordinates of a pixel on im1
    Output: x2, x-coordinates of the pixel on im2
            y2, y-coordinates of the pixel on im2
            
    Hints:
    (1) Given input [x1, x2], use the fundamental matrix to recover the corresponding epipolar line on image2
    (2) Search along this line to check nearby pixel intensity (you can define a search window) to 
        find the best matches
    (3) Use guassian weighting to weight the pixel simlairty

'''
def epipolarCorrespondence(im1, im2, F, x1, y1):
    # Replace pass by your implementation
    window_size = 5
    # x1 = np.flatten(x1)
    # y1 = np.flatten(y1)

 
    # Initializing value of x-axis and y-axis
    # in the range -1 to 1
    x, y = np.meshgrid(np.linspace(-1,1,1+(window_size*2)), np.linspace(-1,1,1+(window_size*2)))
    dst = np.sqrt(x*x+y*y)
    # Initializing sigma and muu
    sigma = 1
    muu = 0.000
    # Calculating Gaussian array
    gauss = np.exp(-( (dst-muu)**2 / ( 2.0 * sigma**2 ) ) )
    gauss = gauss/np.sum(gauss)

    gauss3 = np.stack((gauss,gauss, gauss),axis=2)
    rows_min1 = y1 - window_size     
    rows_max1 = y1 + window_size+1

    col_min1 = x1 - window_size
    col_max1 = x1 + window_size+1

    P1 = [] 
    w1 = im1[rows_min1:rows_max1, col_min1:col_max1]
    
    P1 = np.asarray([x1,y1,1])

    vector = F @ P1

    rows = np.arange(window_size,(im2.shape[0]-window_size))
    cols = (-(vector[1]/vector[0])*rows + (-vector[2]/vector[0])).astype(int)
    
    
    minimum_dist = 100000
    for i in range(len(cols)):
        rows_min2 = rows[i] - window_size
        rows_max2 = rows[i] + window_size + 1

        col_min2 = cols[i] - window_size
        col_max2 = cols[i] + window_size + 1
        w2 = im2[rows_min2:rows_max2, col_min2:col_max2]

        w_err = (w1-w2)*gauss3
        diffd = []

        diff1 = np.linalg.norm(w_err)
        p2 = np.asarray([rows[i],cols[i]])
        if diff1 <=minimum_dist:
            p2 = p2
            y2 = p2[0]
            x2 = p2[1]
            minimum_dist = diff1

    return x2,y2




if __name__ == "__main__":

    correspondence = np.load('../data/some_corresp.npz') # Loading correspondences
    intrinsics = np.load('../data/intrinsics.npz') # Loading the intrinscis of the camera
    K1, K2 = intrinsics['K1'], intrinsics['K2']
    pts1, pts2 = correspondence['pts1'], correspondence['pts2']
    im1 = plt.imread('../data/im1.png')
    im2 = plt.imread('../data/im2.png')


    # ----- TODO -----
    # YOUR CODE HERE
    
    F = eightpoint(pts1, pts2, M=np.max([*im1.shape, *im2.shape]))
        
    x2, y2 = epipolarCorrespondence(im1, im2, F, 119, 217)
    epipolarMatchGUI(im1,im2,F)
    np.savez('../results/q4_1.npz',F,pts1,pts2)
    assert(np.linalg.norm(np.array([x2, y2]) - np.array([118, 181])) < 10)