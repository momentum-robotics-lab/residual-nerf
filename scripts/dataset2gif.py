import argparse
import glob 
import imageio
import os 
import re 

parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str, required=True)
parser.add_argument('--fps', type=int, default=10)
parser.add_argument('--out', type=str, default='out.gif')
parser.add_argument('--idx',type=int,default=0)
args = parser.parse_args()

def natural_stort_key(string):
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)',string)]


if __name__ == '__main__':
    files = sorted(glob.glob(os.path.join(args.dir, '*' + str(args.idx)+'.png')),key=natural_stort_key)

    images = []
    for f in files:
        images.append(imageio.imread(f))
    
    duration = 1.0 / args.fps * len(images)
    imageio.mimsave(args.out, images, duration=duration)
    print('Saved gif to %s' % args.out)