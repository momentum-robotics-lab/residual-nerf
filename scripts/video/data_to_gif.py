import argparse
from natsort import natsorted 
import glob
import os 
import imageio

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='data/vids/vid1.mp4')
parser.add_argument('--output_dir', type=str, default='data_gifs')
args = parser.parse_args()

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

folders = glob.glob(os.path.join(args.data, '*'))
folders = [f for f in folders if os.path.isdir(f)]

for folder in folders:
    data_folder = os.path.join(folder, 'test')
    # Get all the files in the input directory
    files = glob.glob(os.path.join(data_folder, '*.png'))

    # filter out background
    files = natsorted([f for f in files if '-1' not in f])
    #sort based on number in string
    images = [imageio.imread(f) for f in files]
    name = data_folder.split('/')[-2]
    print(name)
    imageio.mimsave(os.path.join(args.output_dir,name+'.gif'), images, duration=50,loop=0)