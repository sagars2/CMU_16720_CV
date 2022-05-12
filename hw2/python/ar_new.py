"""
@author: 16-720A Teaching Staff
@author: Bassam Bikdash
HW2: Augmented Reality with Planar Homographies

ar.py

Due: 3/11/2021
"""
# Import necessary functions
import time
import numpy as np
import skimage
import skimage.io
import multiprocessing as mp
from itertools import repeat
from loadVid import loadVid
import imageio
from opts import get_opts
from ar_helper import composeWarpedImg

def composeFrames(book_frame, src_frame, cv_cover):
    print('Frame started')
    return composeWarpedImg(cv_cover, book_frame, src_frame, opts)

def main():
    opts = get_opts()
    # Write function for Q3.1
    start = time.time()
    # load in necessary data
    src_frames = loadVid('../data/ar_source.mov')
    # Cut the zero padded region and convert to RGB
    src_frames = src_frames[:, 48:-48, :, ::-1]     # [511, 264, 640, 3]

    book_frames = loadVid('../data/book.mov') # [641, 480, 640, 3]
    book_frames = book_frames[:, :, :, ::-1]  # convert to RGB
    cv_cover = skimage.io.imread('../data/cv_cover.jpg')    # [440, 350]

    # crop src_frames so that it has the same aspect ratio as the cv_cover
    src_frames = src_frames[:, :, 214:424, :] # We want 210 pixel width for each frame

    # also pad at the end of src_frames with the beginning of src_frames until number of srcframes equals number of book_frames
    src_frames = np.append(src_frames, src_frames[0:130,:,:,:], axis=0) # [641, 264, 210, 3]

    # use ar_helper.composeWarpedImg to place each src_frame correctly over each book_frame, using cv_cover as a reference for warping
    n_worker = mp.cpu_count()
    with mp.Pool(processes=n_worker) as pool:
        composite_frames = pool.starmap(composeFrames,
                      zip(book_frames, src_frames, repeat(cv_cover)))  # Multiple arguments must be passed as a tuple

    # save composite frames into a video, where composite_frames is a list of image arrays of shape (h, w, 3) obtained above
    imageio.mimwrite('../result/ar.avi', composite_frames, fps=30)
    end = time.time()
    

if __name__ == '__main__':
    main()
