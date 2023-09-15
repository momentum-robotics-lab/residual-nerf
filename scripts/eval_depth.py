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
import copy 
import pandas as pd
# some data (512x512)

# map the normalized data to colors
# image is now RGBA (512x512x4) 
depth_names = ['nerf','dex','ours']

parser = argparse.ArgumentParser()
parser.add_argument('--dir',type=str,default='depth_evaluation')
parser.add_argument('--scale',type=float,default=0.33)
parser.add_argument('--debug',action='store_true')
parser.add_argument('--thresh',type=float,default=1.5)
parser.add_argument('--crop',type=int,default=0)
args = parser.parse_args()


class Depth_time:
    def __init__(self,npz,gt,epoch,dex=False):
        self.epoch = epoch
        self.data_dict = np.load(npz) 
        self.gt_depth = gt
        if dex:
            self.infer_depth = self.data_dict['dex_depth']
        else:
            self.infer_depth = self.data_dict['nerf_depth']
        
        self.time = self.data_dict['time']
        print(self.time)
        self.infer_depth *= (1.0/args.scale)
        # print the attributes of the data
        if args.crop > 0:
            # crop from center
            b, h,w = depth.shape
            self.infer_depth = depth[:,h//2-args.crop:h//2+args.crop,w//2-args.crop:w//2+args.crop]
            
            b, h, w = self.gt_depth.shape
            self.gt_depth = self.gt_depth[:,h//2-args.crop:h//2+args.crop,w//2-args.crop:w//2+args.crop]
        
        self.gt_depth_viz = copy.deepcopy(self.gt_depth[0])

        self.viz_depth = self.infer_depth[0]

        self.infer_depth = self.infer_depth.flatten()
        self.gt_depth = gt.flatten()
        self.mask = self.gt_depth < args.thresh 
        self.infer_depth = self.infer_depth[self.mask]
        self.gt_depth = self.gt_depth[self.mask]
        self.rmse = self.rmse()
        self.mae = self.mae()

    def rmse(self,dex=True):
        # remove values off the table
        rmse = np.sqrt(np.mean((self.gt_depth - self.infer_depth)**2))

        return rmse

    def mae(self):
        mae = np.mean(np.abs(self.gt_depth - self.infer_depth))
        return mae

class Depth:
    def __init__(self,dir,gt,name=None):
        self.name=name 
        print(self.name)
        self.gt = gt
        self.dir = dir 
        self.depth_files = glob.glob(os.path.join(dir,'*.npz'))
        # sort in natural way
        self.depth_files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
        self.load_depths()
    
    def load_depths(self):
        self.depths_epoch = []
        for depth_file in self.depth_files:
            # get file name only
            name = os.path.basename(depth_file)
            # get the digit part of it
            epoch = int(''.join(filter(str.isdigit, name)))
            self.depths_epoch.append(Depth_time(depth_file,self.gt,epoch))
            

class Scene:
    # init
    def __init__(self,scene_dir):
        self.depths = []
        self.scene_dir = scene_dir

        self.depth_dirs = glob.glob(os.path.join(scene_dir,'*'))
        self.depth_dirs = [d for d in self.depth_dirs if os.path.isdir(d)]

        self.gt_depth_file = os.path.join(scene_dir,'gt.npy')
        self.n_depths = len(self.depth_dirs)
        self.depth_names = depth_names
        
        self.load_depths()
        if args.debug:
            self.debug()

    def load_depths(self):
        # load all depths
        
        print('Loading depths for scene: ', self.scene_dir)
        self.gt = np.load(self.gt_depth_file)
        self.gt = self.gt[:,:,:,0]
        
        if args.crop > 0:
            b, h,w = self.gt.shape
            self.gt = self.gt[:,h//2-args.crop:h//2+args.crop,w//2-args.crop:w//2+args.crop]


        for depth_dir in self.depth_dirs:
            
            self.depths.append(Depth(depth_dir,self.gt,name=os.path.basename(depth_dir)))

    def debug(self):
        ax1 = plt.subplot(1,self.n_depths+1,1)
        ax1.imshow(self.depths[0].depths_epoch[-1].gt_depth_viz)
        #title first plot 
        ax1.set_title('Ground Truth')

        for i in range(1,self.n_depths+1):
            ax = plt.subplot(1,self.n_depths+1,i+1)
            ax.imshow(self.depths[i-1].depths_epoch[-1].viz_depth)
            ax.set_title(self.depth_names[i-1])
    
        plt.show()

# find all dirs in args.dir

scenes = glob.glob(os.path.join(args.dir,'*'))
# filter out non dirs
scenes = [s for s in scenes if os.path.isdir(s)]

# check if empty 
scenes = [s for s in scenes if len(os.listdir(s)) > 0]

# check if gt.npy exists
scenes = [s for s in scenes if os.path.exists(os.path.join(s,'gt.npy'))]

scenes = [Scene(s) for s in scenes]

scene_objs = scenes
# put rmse and mae into table 





# 
field_names = ['Scene']

# scene names

scenes_names = ['shelf','clutter','bowl','drink_flat','drink_up','needles_flat','needles_up','scissors_flat','scissors_up','wine_flat','wine_up']


   
table = pt.PrettyTable()
table.field_names = ['Method'] + scenes_names
df = pd.DataFrame(columns=['Method'] + scenes_names)
for depth_name in depth_names:
    row = [depth_name]
    for scene_name in scenes_names:
        found_scene = False
        for scene_obj in scene_objs:
            if scene_name in scene_obj.scene_dir:
                for depth in scene_obj.depths:
                    if depth_name in depth.name:
                        row.append('{:.4f}'.format(depth.depths_epoch[-1].rmse))
                        
                        found_scene = True
                        break
        if not found_scene:
            row.append('N/A')

    table.add_row(row)
    df.loc[len(df)] = row
print("RMSE")
print(table)
print(df.to_latex(index=False))



table = pt.PrettyTable()
table.field_names = ['Method'] + scenes_names
df = pd.DataFrame(columns=['Method'] + scenes_names)

for depth_name in depth_names:
    row = [depth_name]
    for scene_name in scenes_names:
        found_scene = False
        for scene_obj in scene_objs:
            if scene_name in scene_obj.scene_dir:
                for depth in scene_obj.depths:
                    if depth_name in depth.name:
                        row.append('{:.4f}'.format(depth.depths_epoch[-1].mae))
                        found_scene = True
                        break
        if not found_scene:
            row.append('N/A')
    table.add_row(row)
    df.loc[len(df)] = row

print("MAE")
print(table)
print(df.to_latex(index=False))