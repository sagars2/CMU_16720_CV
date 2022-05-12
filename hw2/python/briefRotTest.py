import numpy as np
import cv2
import matplotlib.pyplot as plt
from matchPics import matchPics
from opts import get_opts
import scipy
from helper import plotMatches
from pylab import savefig
#Q2.1.6

img = cv2.imread('../data/cv_cover.jpg')
hist_update = []

def rotTest(img,opts):

    #Read the image and convert to grayscale, if necessary
    for i in range(37):

        #Rotate Image
        rot = scipy.ndimage.rotate(img,10*i,reshape=True)
        
        #Compute features, descriptors and Match features
        matches, locs1, locs2 = matchPics(img, rot, opts)
    
        #Update histogram
        hist_update.append(len(matches))
        
        if i == 0:
            n = plotMatches(img, rot, matches, locs1, locs2)
            # n.savefig('../results/img_0.jpg')
        if i == 3:
            n = plotMatches(img, rot, matches, locs1, locs2)
            # n.savefig('../results/img_3.jpg')
        if i == 15:
           n = plotMatches(img, rot, matches, locs1, locs2)
           # n.savefig('../results/img_15.jpg')
        if i == 27:
            n = plotMatches(img, rot, matches, locs1, locs2)
            # n.savefig('../results/img_27.jpg')
        if i == 36:
            n = plotMatches(img, rot, matches, locs1, locs2)
            # n.savefig('../results/img_36.jpg')
        # hist, bins = np.histogram(hist_update,37,(0,360))
        # print(rot.shape)
        # print(hist_update)
    #Display histogram
    # plt.hist(hist_update)
    return hist_update

if __name__ == "__main__":

    opts = get_opts()
    hist_update = rotTest(img,opts)
    plt.bar(np.arange(0,370,10), hist_update, color='blue', width=7)
    plt.ylabel('No. of matches')
    plt.xlabel('Rotation Angles')
    plt.show()

