import argparse
import glob 
import os 
import re
from natsort import natsorted
import imageio
import cv2

def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)

def read_image(path):
    img =  imageio.imread(path)
    # get step number from path 
    step = int(path.split('/')[-1].split('.')[0].split('_')[-1])
    # txt = 'Step: ' + str(step)
    # show step in the top-right corner of the image
    # cv2.putText(img, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    # print(step)
    return img

def make_gif(images,outpath,duration):
    imageio.mimsave(outpath, images, duration=duration,loop=0)

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, required=True)
args = parser.parse_args()

# Get all the files in the input directory
files = glob.glob(os.path.join(args.input, '*.png'))
nerf_files = natsorted([f for f in files if 'nerf' in f])
dex_files = natsorted([f for f in files if 'dex' in f])
our_files = natsorted([f for f in files if 'our' in f])

nerf_images = [read_image(f) for f in nerf_files]
dex_images = [read_image(f) for f in dex_files]
our_images = [read_image(f) for f in our_files]

duration = 500
make_gif(nerf_images, os.path.join(args.input, 'nerf.gif'), duration)
make_gif(dex_images, os.path.join(args.input, 'dex.gif'), duration)
make_gif(our_images, os.path.join(args.input, 'our.gif'), duration)


