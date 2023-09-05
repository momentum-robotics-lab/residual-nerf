import numpy as np 
import argparse
import imageio
import os 
import numpy as np
import matplotlib.pyplot as plt
import cv2 
# some data (512x512)

# map the normalized data to colors
# image is now RGBA (512x512x4) 

to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)

parser = argparse.ArgumentParser()
parser.add_argument('-i', type=str, default='depth.npy', help='path to depth file')
parser.add_argument('-max', type=float, default=1.0, help='max depth value')
parser.add_argument('-min', type=float, default=0.0, help='min depth value')
parser.add_argument('-auto',action='store_true',help='auto min max')
parser.add_argument('-o',type=str,default='depth.mp4',help='output file name')
parser.add_argument('--debug',action='store_true',help='debug mode')
args = parser.parse_args()

depths = np.load(args.i,allow_pickle=True)

depths_new = []

print(depths.shape)
for i in range(len(depths)):
    # depths_new.append(depths[i][0])

    if args.auto:
        args.max = np.max(depths[i])
        args.min = np.min(depths[i])
        depth = (depths[i]-args.min)/(args.max - args.min)
    else:

        depth = depths[i]
        depth = np.clip(depth,args.min,args.max)
        depth = (depth-args.min)/(args.max - args.min)
    
    
    image = depth
    # image = cv2.applyColorMap(np.uint8(depth*255), cv2.COLORMAP_TURBO)
    # image = depth
    if args.debug:
        plt.imshow(image)
        plt.show()
        exit()
    depths_new.append(image)



imageio.mimwrite(args.o, to8b(depths_new), fps=25, quality=8, macro_block_size=1)