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
args = parser.parse_args()

gt_depth = load_exr(args.gt)
