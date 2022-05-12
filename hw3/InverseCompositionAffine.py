import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import affine_transform, sobel

def InverseCompositionAffine(It, It1, threshold, num_iters):
    """
    :param It: template image
    :param It1: Current image
    :param threshold: if the length of dp is smaller than the threshold, terminate the optimization
    :param num_iters: number of iterations of the optimization
    :return: M: the Affine warp matrix [2x3 numpy array] put your implementation here
    """
    # put your implementation here
    M = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    p = np.zeros(6)
    
    h1 = It.shape[0]
    w1 = It.shape[1]
    h2 = It1.shape[0]
    w2 = It1.shape[1]
    interp_It = RectBivariateSpline(np.arange(h1),np.arange(w1),It)
    interp_It1 = RectBivariateSpline(np.arange(h2),np.arange(w2),It1)

    for i in range(int(num_iters)):
        #Computing b
        It_warp = affine_transform(It, M)

        #Computing A
        grad_col, grad_row = np.meshgrid(np.linspace(0,w1-1,w1),np.linspace(0,h1-1,h1))

        grad_col = grad_col.flatten()
        grad_row = grad_row.flatten()
        
        dx = affine_transform(sobel(It1,1),M)
        dy = affine_transform(sobel(It1,0),M)

        dx = dx.flatten().T
        dy = dy.flatten().T
        
        A = np.array([dx * grad_col, dx * grad_row, dx,
                        dy * grad_col, dy * grad_row, dy]).T
        b = (It-It_warp).flatten()
        del_p = np.linalg.pinv(A.T @ A) @ A.T @ b
        p += del_p
        M[0,0] = 1+p[0]
        M[0,1] = p[1]
        M[0,2] = p[2]
        M[1,0] = p[3]
        M[1,1] = 1+p[4]
        M[1,2] = p[5]

        del_M = np.array([[1+del_p[0], del_p[1], del_p[2]],
        [del_p[3],del_p[4]+1, del_p[5]]])
        M = M @ np.linalg.pinv(del_M)
        if np.linalg.norm(del_p) < threshold:
            break
    return M