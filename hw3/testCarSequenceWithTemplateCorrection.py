import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
from LucasKanade import LucasKanade

# write your script here, we recommend the above libraries for making your animation

parser = argparse.ArgumentParser()
parser.add_argument('--num_iters', type=int, default=1e4, help='number of iterations of Lucas-Kanade')
parser.add_argument('--threshold', type=float, default=1e-2, help='dp threshold of Lucas-Kanade for terminating optimization')
parser.add_argument('--template_threshold', type=float, default=5, help='threshold for determining whether to update template')
args = parser.parse_args()
num_iters = args.num_iters
threshold = args.threshold
template_threshold = args.template_threshold

seq = np.load("../data/carseq.npy")
rect = [59, 116, 145, 151]
rect1 = [59, 116, 145, 151]
height = seq.shape[0]
width = seq.shape[1]
frame_idx = seq.shape[2]
frames = [0,99,199,299,399]

rectangle = [rect]
vid_fr = []
p = np.zeros((2))
rect_og = np.load("../results/carseqrects.npy")
for i in range(frame_idx-1):
    It = seq[:,:,i]
    It1 = seq[:,:,i+1]
    p = LucasKanade(It,It1,rect,threshold,num_iters,p)
    ptotal = np.array((p[0]+rect[0]-rect1[0],p[1]+rect[1]-rect1[1]))
    pstar = LucasKanade(seq[:,:,0],It1,rect1,threshold,num_iters,ptotal)
    delta_p = pstar-ptotal
    
    diff = pstar+[rect[0]-rect1[0],rect[1]-rect1[1]]
    if np.linalg.norm(delta_p) <= template_threshold:
       rect[0] = rect1[0] + pstar[0]
       rect[1] = rect1[1] + pstar[1]
       rect[2] = rect1[2] + pstar[0]
       rect[3] = rect1[3] + pstar[1]
    #    p0 = np.zeros((2))
    else:
        rect[0] += ptotal[0]
        rect[1] += ptotal[1]
        rect[2] += ptotal[0]
        rect[3] += ptotal[1]
        # p0 = p
    rectangle.append(rect)
    bbox = rectangle[i]
    vid = np.zeros((height, width, 3))
    vid[:, :, 0] = It1
    vid[:, :, 1] = It1
    vid[:, :, 2] = It1
    
    #new one
    cv2.rectangle(vid,(int(bbox[0]),int(bbox[1])),(int(bbox[2]),int(bbox[3])),(255,0,0),1)
    #older rectangle
    cv2.rectangle(vid,(int(rect_og[i][0]),int(rect_og[i][1])),(int(rect_og[i][2]),int(rect_og[i][3])),(0,0,255),1)
    
    cv2.imshow('output',vid)
    cv2.waitKey(1)
    vid_fr.append(vid)
    if i%10 == 0:
        print('Iteration:',i)

for f in frames:
    image = (vid_fr[f] * 255).astype(int)
    cv2.imwrite('../results/corrected_car_frame{}.jpg'.format(f),image)

np.save('../results/carseqrects-wcrt.npy',rectangle)
# g = np.load('../data/results/carseqrects-wcrt.npy')
# print(g.shape)
