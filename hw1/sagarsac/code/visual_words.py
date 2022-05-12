import math
import multiprocessing
import os
from copy import copy
from os.path import join

from multiprocessing import Pool
from itertools import repeat

import numpy as np
import pandas as pd
import scipy.ndimage
import skimage.color
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.cluster import KMeans
# from tqdm.autonotebook import tqdm
from sklearn.metrics import confusion_matrix


def extract_filter_responses(opts, img):
    '''
    Extracts the filter responses for the given image.

    [input]
    * opts    : options
    * img    : numpy.ndarray of shape (H,W) or (H,W,3)
    [output]
    * filter_responses: numpy.ndarray of shape (H,W,3F)
    '''
    filter_scales = opts.filter_scales
    # ----- TODO -----
    lab_color_space = skimage.color.rgb2lab(img)
    img_shape = np.shape(img)
    H = img_shape[0]
    W = img_shape[1]
    F = 4*len(filter_scales)
    #grayscale to RGB
    if len(img_shape) < 3:
        img = np.hstack((img.shape,3))
    #3F dimension to RGB
    if img_shape[2] > 3:
        img = img[:,:,0:3]
    filter_responses = np.zeros((H,W,3*F))
    for i in range(len(filter_scales)):
        for j in range(3):
            #Gaussian
            filter_responses[:,:,(12*i)+j] = scipy.ndimage.gaussian_filter(lab_color_space[:,:,j], filter_scales[i])
            #Laplacian of Gaussian
            filter_responses[:,:,(12*i)+j+3] = scipy.ndimage.gaussian_laplace(lab_color_space[:,:,j], filter_scales[i])
            #derivative of Gaussian in x
            filter_responses[:,:,(12*i)+j+6] = scipy.ndimage.gaussian_filter(lab_color_space[:,:,j], filter_scales[i], [1,0])
            #derivative of Gaussian in y
            filter_responses[:,:,(12*i)+j+9] = scipy.ndimage.gaussian_filter(lab_color_space[:,:,j], filter_scales[i], [0,1])            
    return filter_responses
    pass


def compute_dictionary_one_image(opts,args):
    """
    Extracts a random subset of filter responses of an image and save it to disk
    This is a worker function called by compute_dictionary

    Your are free to make your own interface based on how you implement compute_dictionary
    """
    data_dir = opts.data_dir
    feat_dir = opts.feat_dir
    out_dir = opts.out_dir
    K = opts.K
    alpha = opts.alpha
    
    # ----- TODO -----
    # train_files = open(join(data_dir, "train_files.txt")).read().splitlines()
    # img_path = join(opts.data_dir, train_files)
    # img_1 = plt.imread(img_path)/255.
    # filter_responses = extract_filter_responses(opts, img_1)
    # pixels = np.empty((alpha,2))
    # H = filter_responses.shape[0]
    # W = filter_responses.shape[1]
    
    # idx = np.random.randint(filter_responses.shape[0],alpha)
    # idy = np.random.randint(filter_responses.shape[1],alpha)
    
    # pixels = np.stack(idx,idy)
    
    # new_img = filter_responses[pixels[0],pixels[1]]
    
    # np.save("new_dictionary",new_img)
    
    pass


def compute_dictionary(opts, n_worker=8):
    """
    Creates the dictionary of visual words by clustering using k-means.

    [input]
    * opts         : options
    * n_worker     : number of workers to process in parallel

    [saved]
    * dictionary : numpy.ndarray of shape (K,3F)
    """

    data_dir = opts.data_dir
    feat_dir = opts.feat_dir
    out_dir = opts.out_dir
    K = opts.K
    alpha = opts.alpha

    train_files = open(join(data_dir, "train_files.txt")).read().splitlines()
    # ----- TODO -----
    filter_scales = opts.filter_scales
#     #extract the responses:
    fr = np.empty((alpha*len(train_files),36))
    for a in range(len(train_files)):
        
        img_path = join(opts.data_dir, train_files[a])
        img_1 = plt.imread(img_path)/255.
        filter_responses = extract_filter_responses(opts, img_1)
        H = filter_responses.shape[0]
        W = filter_responses.shape[1]
        F = 4*len(filter_scales)
    
        pixels = np.empty((alpha,2))
        for i in range(alpha):
            idx_1 = np.random.randint(H)
            idx_2 = np.random.randint(W)
            pixels[i,0] = idx_1
            pixels[i,1] = idx_2
            
        for i in range(alpha):
            fr[i*a,:] = filter_responses[int(pixels[i,0]),int(pixels[i,1])]
         #run Kmeans
        if a%100 == 0:
            print("Iteration Number: ",a)
    kmeans = KMeans(n_clusters=K).fit(fr)
    dictionary = kmeans.cluster_centers_
    np.save("dictionary",dictionary)
    pass

    # example code snippet to save the dictionary
    # np.save(join(out_dir, 'dictionary.npy'), dictionary)


def get_visual_words(opts, img, dictionary):
    """
    Compute visual words mapping for the given img using the dictionary of visual words.

    [input]
    * opts    : options
    * img    : numpy.ndarray of shape (H,W) or (H,W,3)

    [output]
    * wordmap: numpy.ndarray of shape (H,W)
    """

    # ----- TODO -----
    filter_responses = extract_filter_responses(opts, img)
    H = filter_responses.shape[0]
    W = filter_responses.shape[1]
    
    wordmap = np.empty((H,W))
    for i in range(H):
        for j in range(W):
            pixel_responses = filter_responses[i,j]
            euc_dist = scipy.spatial.distance.cdist(pixel_responses[:,None].T, dictionary,'euclidean')
            wordmap[i,j] = np.argmin(euc_dist)
    return wordmap
    pass
