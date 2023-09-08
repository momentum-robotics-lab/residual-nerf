import numpy as np 
import argparse
import imageio
import os 
import numpy as np
import matplotlib.pyplot as plt
import cv2 
import pyroexr
import glob
import prettytable as pt
# some data (512x512)

# map the normalized data to colors
# image is now RGBA (512x512x4) 


parser = argparse.ArgumentParser()
parser.add_argument('--dir',type=str,default='depth_evaluation')
parser.add_argument('--scale',type=float,default=0.33)
parser.add_argument('--debug',action='store_true')
parser.add_argument('--thresh',type=float,default=1.5)
args = parser.parse_args()

class Depth:
    def __init__(self,depth,gt,name=None):
        self.name=name
        self.infer_depth = depth.flatten()
        self.gt_depth = gt.flatten()
        self.mask = self.gt_depth < args.thresh 
        self.infer_depth = self.infer_depth[self.mask]
        self.gt_depth = self.gt_depth[self.mask]

        self.rmse = self.rmse()
        self.mae = self.mae()

    def rmse(self):
        # remove values off the table
        rmse = np.sqrt(np.mean((self.gt_depth - self.infer_depth)**2))

        return rmse

    def mae(self):
        mae = np.mean(np.abs(self.gt_depth - self.infer_depth))
        return mae


class Scene:
    # init
    def __init__(self,scene_dir):
        self.depths = []
        self.scene_dir = scene_dir
        self.depths_files = glob.glob(os.path.join(scene_dir,'*.npy'))
        self.depths_files = [f for f in self.depths_files if 'gt' not in f]
        self.gt_depth_file = os.path.join(scene_dir,'gt.npy')
        self.n_depths = len(self.depths_files)
        self.depth_names = [os.path.basename(f) for f in self.depths_files]
        
        self.load_depths()
        if args.debug:
            self.debug()

    def load_depths(self):
        # load all depths
        
        print('Loading depths for scene: ', self.scene_dir)
        self.gt = np.load(self.gt_depth_file)
        self.gt = self.gt[:,:,:,0]
        for f in self.depths_files:
            # load depth
            depth = np.load(f)
            # scale depth
            depth *= (1.0/args.scale)
            
            self.depths.append(Depth(depth,self.gt,name=os.path.basename(f)))

    def debug(self):
        ax1 = plt.subplot(1,self.n_depths+1,1)
        ax1.imshow(self.gt[0])
        #title first plot 
        ax1.set_title('Ground Truth')

        for i in range(1,self.n_depths+1):
            print(i)
            ax = plt.subplot(1,self.n_depths+1,i+1)
            ax.imshow(self.depths[i-1].depth[0])
            ax.set_title(self.depth_names[i-1])
        


        plt.show()

# find all dirs in args.dir

scenes = glob.glob(os.path.join(args.dir,'*'))
# filter out non dirs
scenes = [s for s in scenes if os.path.isdir(s)]

scenes = [Scene(s) for s in scenes]

# put rmse and mae into table 

table = pt.PrettyTable()
table.field_names = ['Scene','Model','RMSE','MAE']
for scene in scenes:
    for depth in scene.depths:
        table.add_row([scene.scene_dir,depth.name,depth.rmse,depth.mae])

print(table)

