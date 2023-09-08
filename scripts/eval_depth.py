import numpy as np 
import argparse
import imageio
import os 
import numpy as np
import matplotlib.pyplot as plt
import cv2 
import pyroexr
import glob

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
    def __init__(self,depth,gt):
        self.depth = depth
        self.gt = gt

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
            
            self.depths.append(Depth(depth,self.gt))

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


# # N x H x W
# gt_depth = np.load(args.gt)[:,:,:,0]
# # gt_depth = gt_depth[:,:,:,0]
# # N x H x W
# infer_depth = np.load(args.infer)
# infer_depth *= (1.0/args.scale)
# if args.debug:
#     ax1 = plt.subplot(1,2,1)
#     ax1.imshow(gt_depth[0])
#     ax2 = plt.subplot(1,2,2)
#     ax2.imshow(infer_depth[0])

#     #title first plot 
#     ax1.set_title('Ground Truth')
#     ax2.set_title('Inferred Depth')

#     plt.show()


# # compute error metrics

# # RMSE
# # N x H x W

# # flatten to (N*H*W)
# gt_depth = gt_depth.flatten()
# infer_depth = infer_depth.flatten()

# # remove values off the table
# mask = gt_depth < args.thresh 
# gt_depth = gt_depth[mask]
# infer_depth = infer_depth[mask]

# rmse = np.sqrt(np.mean((gt_depth - infer_depth)**2))

# print('RMSE: ', rmse)