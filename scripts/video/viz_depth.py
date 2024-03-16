import numpy as np 
import argparse
import imageio
import os 
import numpy as np
import matplotlib.pyplot as plt
import cv2 
from natsort import natsorted

# some data (512x512)

# map the normalized data to colors
# image is now RGBA (512x512x4) 

to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)

parser = argparse.ArgumentParser()
parser.add_argument('-i', type=str, default='depth.npy', help='path to depth file')
parser.add_argument('-max', type=float, default=0.7, help='max depth value')
parser.add_argument('-min', type=float, default=0.0, help='min depth value')
parser.add_argument('-auto',action='store_true',help='auto min max')
parser.add_argument('-o',type=str,default='output_depth',help='output folder for imgs')
parser.add_argument('--debug',action='store_true',help='debug mode')
parser.add_argument('--gif_dir',type=str,default='gifs',help='output folder for gifs')
args = parser.parse_args()

depths_data = np.load(args.i,allow_pickle=True)

extension = args.i.split('.')[-1]


if extension == 'npz':
    depths = depths_data['dex_depth']
    rgb_0 = depths_data['rgb'] # H x W x 3

    # save 
    imageio.imwrite('rgb_0.png',rgb_0)
    
elif extension == 'npy':
    depths = depths_data

    #DEBUG
    depths = depths[:10]
    # H x W -> 1 x H x W
    if len(depths.shape) == 2:
        depths = np.expand_dims(depths,axis=0)

    plt.imshow(depths[0],cmap='magma')
    plt.show()

depths_new = []

if os.path.exists(args.o):
    os.system('rm -rf '+args.o)
os.makedirs(args.o)

if not os.path.exists(args.gif_dir):
    os.makedirs(args.gif_dir)

def make_gif(imgs_dir,name='depth.gif'):
    images = []
    for filename in natsorted(os.listdir(imgs_dir)):
        images.append(imageio.imread(os.path.join(imgs_dir,filename)))
    
    fps = 20 
    duration = 1000/fps
    imageio.mimsave(os.path.join(args.gif_dir,name), images,duration=duration)

for i in range(len(depths)):


    depth = depths[i]
    if args.debug:
        plt.imshow(depth,cmap='magma', vmin=args.min,vmax=args.max)
        plt.show()

    depth = np.clip(depth,args.min,args.max)
    depth = (depth-args.min)/(args.max - args.min)
    
    image = depth
    
    plt.clf()
    plt.imshow(image,cmap='magma',vmin=args.min,vmax=args.max)
    

    # turn off outer axes
    plt.axis('off')

    # tight layout
    plt.tight_layout()
    # save image
    plt.savefig(os.path.join(args.o,'{:04d}.png'.format(i)),bbox_inches='tight',pad_inches=0)
    
    
# name based on input folder
name = args.i.split('/')[-3]
make_gif(args.o,name=name+'.gif')

    # plt.show()



