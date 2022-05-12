import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches

import skimage
import skimage.measure
import skimage.color
import skimage.restoration
import skimage.io
import skimage.filters
import skimage.morphology
import skimage.segmentation

from nn import *
from q4 import *
# do not include any more libraries here!
# no opencv, no sklearn, etc!
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

for img in os.listdir('../images'):
    im1 = skimage.img_as_float(skimage.io.imread(os.path.join('../images',img)))
    bboxes, bw = findLetters(im1)

    plt.imshow(bw)
    for bbox in bboxes:
        minr, minc, maxr, maxc = bbox
        rect = matplotlib.patches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                fill=False, edgecolor='red', linewidth=2)
        plt.gca().add_patch(rect)
    plt.show()
    # find the rows using..RANSAC, counting, clustering, etc.
    ##########################
    ##### your code here #####
    ##########################
    bboxes.sort(key=lambda x:x[2])
    rows = 0
    bottom_boundary = bboxes[0][2]
    d = {rows: []}
    for bbox in bboxes:
        minr, minc, maxr, maxc = bbox
        if minr >= bottom_boundary:
            rows+=1
            bottom_boundary = maxr
            d[rows] = []
        d[rows].append(bbox)
    # d_0 = np.asarray(d[0]).flatten()
    # d_1 = np.asarray(d[1]).flatten()
    # d_2 = np.asarray(d[2]).flatten()
    # d = np.hstack((d_0,d_1,d_2)).flatten()

    # crop the bounding boxes
    # note.. before you flatten, transpose the image (that's how the dataset is!)
    # consider doing a square crop, and even using np.pad() to get your images looking more like the dataset
    ##########################
    ##### your code here #####
    ##########################
    images = []
    img = []
    dataset = np.zeros([len(bboxes),1024])
    for k in d.keys():
        total_count = 0
        list_of_boxes = d[k]
        for bbox in list_of_boxes:
            minr, minc, maxr, maxc = bbox
            letter = im1[minr:maxr,minc:maxc]
            l = skimage.morphology.binary_erosion(letter,skimage.morphology.square(10)).astype(float)
            im1 = skimage.transform.resize(l, (32, 32))

            dataset[total_count,:] = im1.flatten()
            total_count += 1

            # im = np.asarray([np.pad(x, (2,), 'constant', constant_values=(np.median(x) ,)) for x in letter])

    # load the weights
    # run the crops through your neural network and print them out
    import pickle
    import string
    letters = np.array([_ for _ in string.ascii_uppercase[:26]] + [str(_) for _ in range(10)])
    params = pickle.load(open('q3_weights.pickle','rb'))
    ##########################
    ##### your code here #####
    ##########################
    h1 = forward(dataset, params, 'layer1',sigmoid)
    probs = forward(h1, params, 'output', softmax)
    pred = np.argmax(probs,axis=1)
    index_chart = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z', 26: '0', 27: '1', 28: '2', 29: '3', 30: '4', 31: '5', 32: '6', 33: '7', 34: '8', 35: '9'}
    for i in pred:
        print('character', index_chart[i])
