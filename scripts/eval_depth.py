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
import shutil
# some data (512x512)
import matplotlib 
matplotlib.rcParams['pdf.fonttype'] = 42

# map the normalized data to colors
# image is now RGBA (512x512x4) 
depth_names = ['trans','nerf','dex','ours']

parser = argparse.ArgumentParser()
parser.add_argument('--dir',type=str,default='depth_evaluation')
parser.add_argument('--scale',type=float,default=0.33)
parser.add_argument('--debug',action='store_true')
parser.add_argument('--thresh',type=float,default=1.5)
parser.add_argument('--crop',type=int,default=150)
parser.add_argument('--min',type=float,default=1.0)
parser.add_argument('--max',type=float,default=1.2)
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
        
        # depth_views_evaluation/drink_up/ours/depth_5.npz
        views = npz.split('/')[-1].split('.')[0].split('_')[-1]
        
        # change time into number of views
        # self.time = self.data_dict['time']
        self.time = views
        self.infer_depth *= (1.0/args.scale)
        # print the attributes of the data
        if args.crop > 0:
            # crop from center
            b, h,w = self.infer_depth.shape
            self.infer_depth = self.infer_depth[:,h//2-args.crop:h//2+args.crop,w//2-args.crop:w//2+args.crop]
            
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


class DummyTransDepth:
    def __init__(self,rmse,mae,viz_depth):
        self.epoch = 0
        self.time = 0 
        self.rmse = rmse
        self.mae = mae
        self.viz_depth = viz_depth

class TransDepth:
    def __init__(self,npy_dir,gt,name='trans'):
        self.name = name
        self.npy_files = glob.glob(os.path.join(npy_dir,'*.npy'))
        self.infer_depth = np.load(self.npy_files[0])

        self.gt_depth = gt
        
        self.infer_depth *= (1.0/args.scale)
        # print the attributes of the data
        if args.crop > 0:
            # crop from center
            b, h,w = self.infer_depth.shape
            self.infer_depth = self.infer_depth[:,h//2-args.crop:h//2+args.crop,w//2-args.crop:w//2+args.crop]
            
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
        self.depths_epoch = [DummyTransDepth(self.rmse,self.mae,self.viz_depth)]


    def rmse(self,dex=True):
        # remove values off the table
        rmse = np.sqrt(np.mean((self.gt_depth - self.infer_depth)**2))

        return rmse

    def mae(self):
        mae = np.mean(np.abs(self.gt_depth - self.infer_depth))
        return mae




class Depth:
    def __init__(self,dir,gt,name=None,dex=False):
        self.name=name 
        self.gt = gt
        self.dir = dir 
        self.depth_files = glob.glob(os.path.join(dir,'*.npz'))
        # sort in natural way
        self.depth_files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
        self.load_depths(dex)
    
    def load_depths(self,dex=False):
        self.depths_epoch = []
        for depth_file in self.depth_files:
            # get file name only
            name = os.path.basename(depth_file)
            # get the digit part of it
            epoch = int(''.join(filter(str.isdigit, name)))

            self.depths_epoch.append(Depth_time(depth_file,self.gt,epoch,dex))
            

class Scene:
    # init
    def __init__(self,scene_dir):
        self.depths = []
        self.scene_dir = scene_dir

        self.figs_dir = os.path.join(scene_dir,'figs')
        if not os.path.exists(self.figs_dir):
            os.mkdir(self.figs_dir)
        else:
            shutil.rmtree(self.figs_dir)
            os.mkdir(self.figs_dir)

        self.depth_dirs = glob.glob(os.path.join(scene_dir,'*'))
        self.depth_dirs = [d for d in self.depth_dirs if os.path.isdir(d)]
        # filter out dirs without npz files
        self.depth_dirs = [d for d in self.depth_dirs if len(glob.glob(os.path.join(d,'*.npz'))) > 0 or len(glob.glob(os.path.join(d,'*.npy'))) > 0 ]

        self.gt_depth_file = os.path.join(scene_dir,'gt.npy')
        self.n_depths = None
        self.depth_names = []
        
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
            if 'dex' in depth_dir:
                self.depths.append(Depth(depth_dir,self.gt,name='dex',dex=True))
                self.depth_names.append('dex')
                self.depths.append(Depth(depth_dir,self.gt,name='nerf',dex=False))
                self.depth_names.append('nerf')
            elif 'trans' in depth_dir:
                self.depths.append(TransDepth(depth_dir,self.gt))
                self.depth_names.append('trans')
            else:
                self.depths.append(Depth(depth_dir,self.gt,name=os.path.basename(depth_dir),dex=True))
                self.depth_names.append(os.path.basename(depth_dir))

        self.n_depths = len(self.depths)
        self.save_depth_imgs()
        self.gen_plots()
        

    def label_name(self,name):
        if name == 'dex':
            return 'Dex'
        elif name == 'nerf':
            return 'NeRF'
        elif name == 'ours':
            return 'Residual-NeRF'
        else:
            return None

    def gen_plots(self):
        # setup plot 
        fontsize = 25
        #set size of figure
        plt.gcf().set_size_inches(10, 2)
        # set fontsize of ticks
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        plt.ylim(0,1.0)

        plt.xlim(0,10000) # 10k steps
         # rmse over time 
        for i in range(self.n_depths):
            label = self.label_name(self.depth_names[i])
            if label is not None:
                plt.plot([d.epoch for d in self.depths[i].depths_epoch],[d.rmse for d in self.depths[i].depths_epoch],label=label)
        
        # plt.legend(fontsize=fontsize)
        # legend on top of figure horizontally centered
        plt.legend(fontsize=fontsize,loc='upper center', bbox_to_anchor=(0.5, 1.45),ncol=3)
        plt.grid()
        plt.xlabel('Epoch',fontsize=fontsize)
        plt.ylabel('RMSE',fontsize=fontsize)
        plt.savefig(os.path.join(self.figs_dir,'rmse_epoch.pdf'),bbox_inches='tight')

        plt.clf()
        plt.xlim(0,10000) # 10k steps
        plt.ylim(0,1.0)
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)

        # mae over time
        for i in range(self.n_depths):
            label = self.label_name(self.depth_names[i])
            if label is not None:
                plt.plot([d.epoch for d in self.depths[i].depths_epoch],[d.mae for d in self.depths[i].depths_epoch],label=label)
        plt.legend(fontsize=fontsize,loc='upper center', bbox_to_anchor=(0.5, 1.5),ncol=3)
        plt.grid()
        plt.xlabel('Epoch',fontsize=fontsize)
        plt.ylabel('MAE',fontsize=fontsize)
        plt.savefig(os.path.join(self.figs_dir,'mae_epoch.pdf'),bbox_inches='tight')
        plt.clf()

        # plt.xlim(0,120)
        # plt.ylim(0,1.0)
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)

        for i in range(self.n_depths):
            label = self.label_name(self.depth_names[i])
            if label is not None:
                plt.plot([d.time for d in self.depths[i].depths_epoch],[d.rmse for d in self.depths[i].depths_epoch],label=label)
        plt.legend(fontsize=fontsize,loc='upper center', bbox_to_anchor=(0.5, 1.5),ncol=3)
        plt.grid()
        # plt.xlabel('Time [s]',fontsize=fontsize)
        plt.xlabel('Views',fontsize=fontsize)
        plt.ylabel('RMSE',fontsize=fontsize)
        plt.gca().invert_xaxis()
        plt.savefig(os.path.join(self.figs_dir,'rmse_time.pdf'),bbox_inches='tight')
        plt.clf()

        # plt.xlim(0,120)
        # plt.ylim(0,1.0)
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        # mae over time
        for i in range(self.n_depths):
            label = self.label_name(self.depth_names[i])
            if label is not None:
                plt.plot([d.time for d in self.depths[i].depths_epoch],[d.mae for d in self.depths[i].depths_epoch],label=label)
        plt.legend(fontsize=fontsize,loc='upper center', bbox_to_anchor=(0.5, 1.5),ncol=3)
        plt.grid()
        # plt.xlabel('Time [s]',fontsize=fontsize)
        plt.xlabel('Views',fontsize=fontsize)
        plt.ylabel('MAE',fontsize=fontsize)
        # reverse x-axis
        plt.gca().invert_xaxis()
        plt.savefig(os.path.join(self.figs_dir,'mae_time.pdf'),bbox_inches='tight')
        plt.clf()

    def debug(self):

        # image debug
        ax1 = plt.subplot(1,self.n_depths+1,1)
        ax1.imshow(self.depths[0].depths_epoch[-1].gt_depth_viz)
        #title first plot 
        ax1.set_title('Ground Truth')
        
        for i in range(1,self.n_depths+1):
            print(i)
            # name
            print(self.depths[i-1].name)
            ax = plt.subplot(1,self.n_depths+1,i+1)
            ax.imshow(self.depths[i-1].depths_epoch[-1].viz_depth)
            ax.set_title(self.depth_names[i-1])
    
        plt.show()
    
    def save_depth_imgs(self):
        for i in range(self.n_depths):
            for j in range(len(self.depths[i].depths_epoch)):
                name = os.path.join(self.figs_dir,'{}_{}.png'.format(self.depth_names[i],self.depths[i].depths_epoch[j].epoch))
                image = self.depths[i].depths_epoch[j].viz_depth
                image = np.clip(image,args.min,args.max)
                image = (image-args.min)/(args.max - args.min)

                plt.clf()
                plt.imshow(image,cmap='magma',vmin=0.0,vmax=1.0)
                

                # turn off outer axes
                plt.axis('off')

                # tight layout
                plt.tight_layout()

                # save image
                plt.savefig(name,bbox_inches='tight',pad_inches=0)
                plt.clf()
       


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
                        # round to 4 decimal places
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