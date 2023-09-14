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


parser = argparse.ArgumentParser()
parser.add_argument('--dir',type=str,default='depth_evaluation')
parser.add_argument('--scale',type=float,default=0.33)
parser.add_argument('--debug',action='store_true')
parser.add_argument('--thresh',type=float,default=1.5)
parser.add_argument('--crop',type=int,default=0)
args = parser.parse_args()

class Depth:
    def __init__(self,depth,gt,name=None):
        self.name=name[:-4]
        

        if args.crop > 0:
            # crop from center
            b, h,w = depth.shape
            self.infer_depth = depth[:,h//2-args.crop:h//2+args.crop,w//2-args.crop:w//2+args.crop]
            
        self.viz_depth = self.infer_depth[0]

        self.infer_depth = self.infer_depth.flatten()
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
        
        if args.crop > 0:
            b, h,w = self.gt.shape
            self.gt = self.gt[:,h//2-args.crop:h//2+args.crop,w//2-args.crop:w//2+args.crop]


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
            ax = plt.subplot(1,self.n_depths+1,i+1)
            ax.imshow(self.depths[i-1].viz_depth)
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

scene_objs = copy.deepcopy(scenes)
del scenes
# put rmse and mae into table 




depth_names = ['nerf','dex','ours']
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
                        row.append('{:.4f}'.format(depth.rmse))
                        
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
                        row.append('{:.4f}'.format(depth.mae))
                        found_scene = True
                        break
        if not found_scene:
            row.append('N/A')
    table.add_row(row)
    df.loc[len(df)] = row

print("MAE")
print(table)
print(df.to_latex(index=False))