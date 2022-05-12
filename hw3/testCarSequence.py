import argparse
from tkinter import image_names
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from LucasKanade import LucasKanade
from PIL import Image, ImageDraw
import cv2

# write your script here, we recommend the above libraries for making your animation
parser = argparse.ArgumentParser()
#default --num_iters = 1e4
parser.add_argument('--num_iters', type=int, default=1e4, help='number of iterations of Lucas-Kanade')
parser.add_argument('--threshold', type=float, default=1e-2, help='dp threshold of Lucas-Kanade for terminating optimization')
args = parser.parse_args()
num_iters = args.num_iters
threshold = args.threshold

seq = np.load('../data/carseq.npy')
rect = [59, 116, 145, 151]
height = seq.shape[0]
width = seq.shape[1]
frame_idx = seq.shape[2]
frames = [0,99,199,299,399]
rectangle = np.zeros((frame_idx,4))
vid_fr = []
p = np.zeros((2))
for i in range(frame_idx-1):
    It = seq[:,:,i]
    It1 = seq[:,:,i+1]
    p = LucasKanade(It,It1,rect,threshold,num_iters,p)
    rect[0] += p[0]
    rect[1] += p[1]
    rect[2] += p[0]
    rect[3] += p[1]

    rectangle[i,:] = rect
    # vid = seq[:,:,i]
    bbox = rectangle[i]
    # output = cv2.rectangle(vid,bbox)

    # img = cv2.imshow(vid)
    # cv2.waitKey(2)
    vid = np.zeros((height, width, 3))
    vid[:, :, 0] = It1
    vid[:, :, 1] = It1
    vid[:, :, 2] = It1

    output = cv2.rectangle(vid,(int(bbox[0]),int(bbox[1])),(int(bbox[2]),int(bbox[3])),(0,0,255),2)
    cv2.imshow('output_img',output)
    cv2.waitKey(2)
    vid_fr.append(output)
    if i%10 == 0:
        print('Iteration:',i)

for f in frames:
    image = (vid_fr[f] * 255).astype(int)
    cv2.imwrite('../results/pltcar_frame_{}.jpg'.format(f),image)

np.save('../results/carseqrects.npy',rectangle)

g = np.load('../results/carseqrects.npy')
# plt.imshow(g)
print(g.shape)
