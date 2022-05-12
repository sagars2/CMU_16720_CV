import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from LucasKanade import LucasKanade
import cv2

# write your script here, we recommend the above libraries for making your animation

parser = argparse.ArgumentParser()
parser.add_argument('--num_iters', type=int, default=1e4, help='number of iterations of Lucas-Kanade')
parser.add_argument('--threshold', type=float, default=1e-2, help='dp threshold of Lucas-Kanade for terminating optimization')
args = parser.parse_args()
num_iters = args.num_iters
threshold = args.threshold
    
seq = (np.load("../data/girlseq.npy"))/255
rect = [280, 152, 330, 318]
height = seq.shape[0]
width = seq.shape[1]
frame_idx = seq.shape[-1]
frames = [0,19,39,59,79]

p = np.zeros((2))
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
    bbox = rectangle[i]

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
    cv2.imwrite('../results/pltgirl_frame_{}.jpg'.format(f),image)

np.save('../results/girlseqrects.npy',np.array(rectangle))

g = np.load('../results/girlseqrects.npy')
print(g.shape)