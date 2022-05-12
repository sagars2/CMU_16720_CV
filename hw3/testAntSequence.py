import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
from SubtractDominantMotion import SubtractDominantMotion

# write your script here, we recommend the above libraries for making your animation

parser = argparse.ArgumentParser()
parser.add_argument('--num_iters', type=int, default=1e3, help='number of iterations of Lucas-Kanade')
parser.add_argument('--threshold', type=float, default=1e-2, help='dp threshold of Lucas-Kanade for terminating optimization')
parser.add_argument('--tolerance', type=float, default=0.2, help='binary threshold of intensity difference when computing the mask')
args = parser.parse_args()
num_iters = args.num_iters
threshold = args.threshold
tolerance = args.tolerance

seq = np.load('../data/antseq.npy')
height = seq.shape[0]
width = seq.shape[1]
frame_idx = seq.shape[-1]
frames = [29,59,89,119]
rectangle = np.zeros((frame_idx,4))
vid_fr = []
for i in range(frame_idx-1):
    It = seq[:,:,i]
    It1 = seq[:,:,i+1]    
    mask = SubtractDominantMotion(It,It1,threshold,num_iters,tolerance)
    if i%10 == 0:
        print('Iteration:',i)
    if i in frames:
        fig, ax = plt.subplots()
        ax.imshow(It1, cmap='gray')
        ax.imshow(np.ma.masked_array(mask, np.invert(mask)), alpha=1, cmap='winter')
        plt.savefig('{}_{}.jpg'.format('../results/ant_inverse',i))
        plt.axis('off')







