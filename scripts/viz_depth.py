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
parser.add_argument('-o',type=str,default='output_depth',help='output folder for imgs')
parser.add_argument('--debug',action='store_true',help='debug mode')
args = parser.parse_args()

depths = np.load(args.i,allow_pickle=True)

depths_new = []

if os.path.exists(args.o):
    os.system('rm -rf '+args.o)
os.makedirs(args.o)

for i in range(len(depths)):


    depth = depths[i]
    depth = np.clip(depth,args.min,args.max)
    depth = (depth-args.min)/(args.max - args.min)
    
    
    image = depth
    plt.clf()
    plt.imshow(image,cmap='magma')

    # turn off outer axes
    plt.axis('off')

    # tight layout
    plt.tight_layout()


    # save image
    plt.savefig(os.path.join(args.o,'{:04d}.png'.format(i)),bbox_inches='tight',pad_inches=0)

    # plt.show()



