import numpy as np
import cv2
import skimage.io 
import skimage.color
from opts import get_opts
from planarH import computeH_ransac
from planarH import compositeH
from matchPics import matchPics
import matplotlib.pyplot as plt

# Import necessary functions

# Q2.2.4
cv_cover = cv2.imread('../data/cv_cover.jpg')
cv_desk = cv2.imread('../data/cv_desk.png')
hp_cover = cv2.imread('../data/hp_cover.jpg')

hp_resize = cv2.resize(hp_cover,(cv_cover.shape[1],cv_cover.shape[0]))

def warpImage(opts):

    pass

if __name__ == "__main__":
    opts = get_opts()
    matches, locs1, locs2 = matchPics(cv_desk,cv_cover,opts)
    best_H2to1, inliers = computeH_ransac(locs2[matches[:,1]],locs1[matches[:,0]],opts)
    
    composite_img = compositeH(best_H2to1, hp_resize, cv_desk)
    cv2.imwrite('../results/hp_new.jpg',composite_img)

    
