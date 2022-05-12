import numpy as np
from scipy.interpolate import RectBivariateSpline

def LucasKanade(It, It1, rect, threshold, num_iters, p0=np.zeros(2)):
    """
    :param It: template image
    :param It1: Current image
    :param rect: Current position of the car (top left, bot right coordinates)
    :param threshold: if the length of dp is smaller than the threshold, terminate the optimization
    :param num_iters: number of iterations of the optimization
    :param p0: Initial movement vector [dp_x0, dp_y0]
    :return: p: movement vector [dp_x, dp_y]
    """
    p = p0
    # Adding a threshold value below which delta p should be
    # Put your implementation here
    h1 = It.shape[0]
    w1 = It.shape[1]
    h2 = It1.shape[0]
    w2 = It1.shape[1]

    x0 = rect[0]
    y0 = rect[1]

    x1 = rect[2]
    y1 = rect[3]

    interp_It = RectBivariateSpline(np.arange(h1),np.arange(w1),It)
    interp_It1 = RectBivariateSpline(np.arange(h2),np.arange(w2),It1)
    
    columns,rows = np.meshgrid(np.linspace(x0,x1,87),np.linspace(y0,y1,36))
    
    var1 = int((y1-y0)+1)
    var2 = int((x1-x0)+1)

    pixel = interp_It.ev(rows,columns)
    for i in range(int(num_iters)):

        warped_rows = np.linspace(p[1]+y0,p[1]+y1, 36)
        warped_col = np.linspace(p[0]+x0,p[0]+x1, 87)
        grad_col, grad_row = np.meshgrid(warped_col,warped_rows)

        dx = interp_It1.ev(grad_row,grad_col,dx=1,dy=0)
        dy = interp_It1.ev(grad_row,grad_col,dx=0,dy=1)

        A = np.vstack((dy.flatten(),dx.flatten())).T
        
        pixel_warped = interp_It1.ev(grad_row,grad_col)
        b = (pixel.flatten()-pixel_warped.flatten())

        del_p = np.linalg.pinv(A.T @ A) @ A.T @ b
        p += del_p
        if np.linalg.norm(del_p) < threshold:
            break
    return p

