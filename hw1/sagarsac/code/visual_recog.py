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
import visual_words


def get_feature_from_wordmap(opts, wordmap):
    '''
    Compute histogram of visual words.

    [input]
    * opts      : options
    * wordmap   : numpy.ndarray of shape (H,W)

    [output]
    * hist: numpy.ndarray of shape (K)
    '''

    K = opts.K
    # ----- TODO -----
    H = wordmap.shape[0]
    W = wordmap.shape[1]
    hist = np.histogram(wordmap,K,range=(0.,K-1),density=True)[0]
    return hist
    pass


def get_feature_from_wordmap_SPM(opts, wordmap):
    """
    Compute histogram of visual words using spatial pyramid matching.

    [input]
    * opts      : options
    * wordmap   : numpy.ndarray of shape (H,W)

    [output]
    * hist_all: numpy.ndarray of shape (K*(4^L-1)/3)=10
    """
    K = opts.K
    L = opts.L 
    # ----- TODO -----
    H = wordmap.shape[0]
    W = wordmap.shape[1]
    w_hist = []
    for l in range(L+1):
        if l > 1:
            weight = 2**(l-L-1)
        else:
            weight = 2**(-L)
        w_hist.append(weight)
    w_hist = np.flip(w_hist) 
#     print("w_hist shape", np.shape(w_hist))
    hist_new = np.array([])
    
    
    for i in range(L+1):
        image_size = (2**L)-i
        patch_h = int(H/image_size)
        patch_w = int(W/image_size)
        layer = []
        for j in range(image_size):
            for g in range(image_size):
                cell = wordmap[(patch_h*g):((patch_h*g)+patch_h),(patch_w*j):((patch_w*j)+patch_w)]
                layer.append(cell)
        
#         print(np.shape(layer))
        for cell in layer:
#             print(cell)
            hist = np.histogram(cell,K,range=(0.,K-1))[0]*w_hist[i]
#             print(hist)
#             print("Hist_b4: ",hist.shape)
            hist_new = np.concatenate((hist_new,hist))
#             print("After",hist_new.shape)

    hist_all = (hist_new/hist_new.sum())
    return hist_all
    pass


def get_image_feature(opts, img_path, dictionary):
    """
    Extracts the spatial pyramid matching feature.

    [input]
    * opts      : options
    * img_path  : path of image file to read
    * dictionary: numpy.ndarray of shape (K, 3F)


    [output]
    * feature: numpy.ndarray of shape (K)
    """

    # ----- TODO -----
    img = plt.imread(img_path)/255.
    img = img[:,:,:3]
    wordmap = visual_words.get_visual_words(opts,img,dictionary)
    feature = get_feature_from_wordmap_SPM(opts,wordmap)

    return feature


    pass


def build_recognition_system(opts, n_worker=8):
    """
    Creates a trained recognition system by generating training features from all training images.

    [input]
    * opts        : options
    * n_worker  : number of workers to process in parallel

    [saved]
    * features: numpy.ndarray of shape (N,M)
    * labels: numpy.ndarray of shape (N)
    * dictionary: numpy.ndarray of shape (K,3F)
    * SPM_layer_num: number of spatial pyramid layers
    """

    data_dir = opts.data_dir
    out_dir = opts.out_dir
    SPM_layer_num = opts.L

    train_files = open(join(data_dir, "train_files.txt")).read().splitlines()
    train_labels = np.loadtxt(join(data_dir, "train_labels.txt"), np.int32)
    dictionary = np.load(join(out_dir, "dictionary.npy"))
    # ----- TODO -----
    feature = []
    a=0
    # for i in range(len(train_files)):
    #     file_path = os.path.join(data_dir,train_files[i])
    #     features = Pool(processes=n_worker).starmap(get_image_feature,zip(repeat(opts), file_path, repeat(dictionary)))
    #     feature.append(features[i])
    #     if a%100 == 0:
    #         print("Iteration Number: ",a)

    for img_path in train_files:
        a +=1
        img_path = join(opts.data_dir, img_path)
        features = get_image_feature(opts,img_path,dictionary)
        feature.append(features)
        if a%100 == 0:
            print("Iteration Number: ",a)
#     print(feature.shape)
    # example code snippet to save the learned system
    np.savez_compressed(join(out_dir, 'trained_system.npz'),
        features=np.array(feature),
        labels=train_labels,
        dictionary=dictionary,
        SPM_layer_num=SPM_layer_num,
    )


def distance_to_set(word_hist, histograms):
#     print("Word_Hist shape: ",word_hist.shape)
#     print("Histogram shape: ",histograms.shape)
    """
    Compute the distance between a histogram of visual words with all training image histograms.

    [input]
    * word_hist: numpy.ndarray of shape (K)
    * histograms: numpy.ndarray of shape (N,K)

    [output]
    * dists: numpy.ndarray of shape (N)
    """

    # ----- TODO -----
    dists = []
    intersection = np.minimum(histograms, word_hist)
    dists = np.sum(intersection,1)
    
#     for i in range(histograms.shape[0]):
#         h_current = histograms[i]
#         intersection = np.minimum(word_hist,h_current)
#         distance = np.sum(intersection)
#         dists.append(distance)
    return np.array(dists)
    pass


def evaluate_recognition_system(opts, n_worker=8):
    """
    Evaluates the recognition system for all test images and returns the confusion matrix.

    [input]
    * opts        : options
    * n_worker  : number of workers to process in parallel

    [output]
    * conf: numpy.ndarray of shape (8,8)
    * accuracy: accuracy of the evaluated system
    """

    data_dir = opts.data_dir
    out_dir = opts.out_dir

    trained_system = np.load(join(out_dir, "trained_system.npz"))
    dictionary = trained_system["dictionary"]
#     print(np.shape(dictionary))
    # using the stored options in the trained system instead of opts.py
    test_opts = copy(opts)
    test_opts.K = dictionary.shape[0]
    test_opts.L = trained_system["SPM_layer_num"]

    test_files = open(join(data_dir, "test_files.txt")).read().splitlines()
    test_labels = np.loadtxt(join(data_dir, "test_labels.txt"), np.int32)

#     print("Length of test files: ",len(test_files))
    
    # ----- TODO -----
    conf = np.zeros((8,8))
    feature_label = trained_system["features"]
#     print("Feature Label shape: ",feature_label.shape)
    training_labels = trained_system["labels"]
#     print("Training Labels size",training_labels.shape)
    
    for i in range(len(test_files)):
#         if test_labels[i] == 0:
#             continue
        file_path = os.path.join(data_dir,test_files[i])
        feat = get_image_feature(opts,file_path,dictionary)
#         print("Feat shape: ",np.shape(feat))
        
        dist = distance_to_set(feat,feature_label)
#         print(np.shape(dist))
        
        #Predicted Label for every i:
        label_pred = int(training_labels[np.argmax(dist)])
#         print("Predicted Label: ",label_pred)
        
        
        #Actual Label for every i:
        label_actual = test_labels[i]
#         print("Actual Label: ",label_actual)
        
        #Confusion Matrix
        conf[label_actual,label_pred] += 1
        
        
        if i%10 ==0:
            print("Number of Iterations: ",i)
    accuracy = (np.trace(conf)/np.sum(conf))*100
    return conf, accuracy    
    pass

# def compute_IDF(opts, n_worker=1):
#     # YOUR CODE HERE
#     pass

# def evaluate_recognition_System_IDF(opts, n_worker=1):
#     # YOUR CODE HERE
#     pass