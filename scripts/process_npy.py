import numpy as np
import argparse 
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('-i', type=str, default='depth.npy', help='path to depth file')
parser.add_argument('-crop', type=int, default=0, help='crop size')
parser.add_argument('-scale', type=float, default=0.33, help='scale factor')
parser.add_argument('-o',type=str,default='output_depth.npy',help='output folder for imgs')
args = parser.parse_args()

depths_data = np.load(args.i)

# crop from center
if args.crop > 0:
    h,w = depths_data.shape
    h_start = h//2 - args.crop//2
    w_start = w//2 - args.crop//2
    depths_data = depths_data[h_start:h_start+args.crop,w_start:w_start+args.crop]

depths_data *= (1.0/args.scale)
plt.imshow(depths_data)
plt.show()

np.save(args.o,depths_data)