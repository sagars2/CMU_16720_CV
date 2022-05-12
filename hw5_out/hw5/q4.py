import numpy as np

import skimage
import skimage.measure
import skimage.color
import skimage.restoration
import skimage.filters
import skimage.morphology
import skimage.segmentation

# takes a color image
# returns a list of bounding boxes and black_and_white image
def findLetters(image):
    bboxes = []
    bw = None
    # insert processing in here
    # one idea estimate noise -> denoise -> greyscale -> threshold -> morphology -> label -> skip small boxes 
    # this can be 10 to 15 lines of code using skimage functions

    ##########################
    ##### your code here #####
    ##########################
    #denoise
    img_denoise = skimage.restoration.denoise_bilateral(image, multichannel=True)
    #greyscale
    img_grey = skimage.color.rgb2gray(img_denoise)
    #threshold
    img_threshold = skimage.filters.threshold_otsu(img_grey)
    #morphology
    bw = skimage.morphology.closing(img_grey<img_threshold, skimage.morphology.square(10))
    bw = skimage.segmentation.clear_border(bw)
    #label
    img_label = skimage.morphology.label(bw,connectivity=2)
    region = skimage.measure.regionprops(img_label)
    for r in region:
        if r.area > 200:
            box = r.bbox
            box = np.asarray(box) + [-10,-10,10,10]

            bboxes.append(box) 
    bw = 1-bw

    return bboxes, bw