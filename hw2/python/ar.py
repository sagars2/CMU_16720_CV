import time
import numpy as np
import cv2
import multiprocessing as mp
from itertools import repeat
import multiprocessing as mp
from itertools import repeat
from opts import get_opts
import skimage.io
import matplotlib.pyplot as plt

#Import necessary functions
from helper import loadVid
from helper import plotMatches
from planarH import computeH_ransac
from planarH import compositeH
from matchPics import matchPics
import imageio
       

start = time.time() 
def frame_gen(opts,book,ar_vid,cv_cover):
    # print('frame time')
    # print(book.shape)
    matches, locs1, locs2 = matchPics(np.array(book),np.array(cv_cover),opts)
    best_H2to1, inliers = computeH_ransac(locs2[matches[:,1]],locs1[matches[:,0]],opts)
    composite_img = compositeH(best_H2to1, ar_vid, book)
    # print('aaaa')
    return composite_img

if __name__== '__main__':
    opts = get_opts()

#Write script for Q3.1
#Load all required images and videos
    book = loadVid('../data/book.mov')
    ar_vid = loadVid('../data/ar_source.mov')
    cv_cover = cv2.imread('../data/cv_cover.jpg')


    #compute the aspect ratio:
    aspect_ratio = cv_cover.shape[1]/cv_cover.shape[0]

    #Removing black space from the ar_vid
    pixel_reject = np.where(np.sum(np.sum(ar_vid[1,:,:,:] == 0, 1),1) > 700)
    video_frames = np.delete(ar_vid,pixel_reject,axis=1)

    #computing dimensions of the book
    book_height = book.shape[1]
    book_width = book.shape[0]

    #Since we need the video to be in the center of the book\
    video_center = int(book_width/2)

    #we need to now crop the video such that it is in the center

    video_frames = video_frames[:,:,130:510,:]

    #Multiprocessing script \
        
    # output = cv2.VideoWriter('../results/ar.avi',fourcc = cv2.VideoWriter_fourcc('X','V','I','D'),fps=30,)
               
        #Write out the video and store them     
    end = time.time()
    n_worker = mp.cpu_count()
    with mp.Pool(processes=n_worker) as pool:
        frames = pool.starmap(frame_gen, zip(repeat(opts),repeat(book), repeat(video_frames), repeat(cv_cover)))
    imageio.mimwrite('../result/ar.avi', frames, fps=30) 
    


    