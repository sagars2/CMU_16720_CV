import numpy as np
from scipy.ndimage.morphology import binary_erosion, binary_dilation
from scipy.ndimage import affine_transform
from LucasKanadeAffine import LucasKanadeAffine
from InverseCompositionAffine import InverseCompositionAffine

def SubtractDominantMotion(image1, image2, threshold, num_iters, tolerance):
    """
    :param image1: Images at time t
    :param image2: Images at time t+1
    :param threshold: used for LucasKanadeAffine
    :param num_iters: used for LucasKanadeAffine
    :param tolerance: binary threshold of intensity difference when computing the mask
    :return: mask: [nxm]
    """
    
    # put your implementation here
    mask = np.ones(image1.shape, dtype=bool)
    # M = LucasKanadeAffine(image1, image2, threshold, num_iters)
    M = InverseCompositionAffine(image1,image2,threshold,num_iters)

    It_warp = affine_transform(image1,M)
    err = np.absolute(It_warp-image2)
    mask = err>tolerance
    mask = binary_dilation(mask,iterations = 5)
    mask = binary_erosion(mask,iterations = 4)

    return mask
