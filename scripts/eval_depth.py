import numpy as np 
import argparse
import imageio
import os 
import numpy as np
import matplotlib.pyplot as plt
import cv2 
import pyroexr

# some data (512x512)

# map the normalized data to colors
# image is now RGBA (512x512x4) 

def load_exr(path):
    exr = pyroexr.load(path)
    shape = exr.channels()['B'].shape
    img = np.zeros((shape[0],shape[1],3))
    img[:,:,0] = exr.channels()['R']
    img[:,:,1] = exr.channels()['G']
    img[:,:,2] = exr.channels()['B']

    return img



to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)

parser = argparse.ArgumentParser()
parser.add_argument('--infer', type=str, default='depth.npy', help='path to depth file')
parser.add_argument('--gt', type=str, default='gt.npy', help='path to depth file')
parser.add_argument('--scale',type=float,default=0.33)
parser.add_argument('--debug',action='store_true')
parser.add_argument('--thresh',type=float,default=1.5)
args = parser.parse_args()

# gt_depth = load_exr(args.gt)
# N x H x W
gt_depth = np.load(args.gt)[:,:,:,0]
# gt_depth = gt_depth[:,:,:,0]
# N x H x W
infer_depth = np.load(args.infer)
infer_depth *= (1.0/args.scale)
if args.debug:
    ax1 = plt.subplot(1,2,1)
    ax1.imshow(gt_depth[0])
    ax2 = plt.subplot(1,2,2)
    ax2.imshow(infer_depth[0])

    #title first plot 
    ax1.set_title('Ground Truth')
    ax2.set_title('Inferred Depth')

    plt.show()


# compute error metrics

# RMSE
# N x H x W

# flatten to (N*H*W)
gt_depth = gt_depth.flatten()
infer_depth = infer_depth.flatten()

# remove values off the table
mask = gt_depth < args.thresh 
gt_depth = gt_depth[mask]
infer_depth = infer_depth[mask]

rmse = np.sqrt(np.mean((gt_depth - infer_depth)**2))

print('RMSE: ', rmse)