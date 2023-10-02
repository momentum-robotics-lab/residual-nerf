import sys 
sys.path.append('..')
import numpy as np
from util.camera_pose_visualizer import CameraPoseVisualizer
import json 
import os  
import argparse
import matplotlib.pyplot as plt

visualizer = None 

class ComputeRotation:
    def __init__(self, origin, vector_x, vector_y,plane):
        self.vector_x = vector_x/np.linalg.norm(vector_x)
        self.vector_y = vector_y/np.linalg.norm(vector_y)
        self.vector_z = np.cross(vector_x,vector_y)
        self.vector_z = self.vector_z/np.linalg.norm(self.vector_z)
        self.origin = origin
        self.plane = plane

    def transform_plane(self):
        t = np.eye(4)

        #shift to origin 
        

        # find the right rotation matrix

        R = np.linalg.inv(np.stack((self.vector_x,self.vector_y,self.vector_z),axis=1))
        print('R: ',R)
        t[:3,:3] = R
        t[:3,3] = -R@self.origin

        self.new_plane = np.dot(self.plane,t)
        return t


class Processor:
    def __init__(self,args):
        self.args = args
        self.load_config()
        self.load_poses()
        self.process()
        self.save_json()
        self.visualize_new_poses()

    def load_config(self):
        with open(self.args.cfg) as f:
            self.cfg = json.load(f)
        
        self.origin_colmap = np.array(self.cfg['origin'])
        self.x_colmap = np.array(self.cfg['x']) - self.origin_colmap
        self.y_colmap = np.array(self.cfg['y']) - self.origin_colmap
        self.x_colmap_norm = np.linalg.norm(self.x_colmap)
        self.y_colmap_norm = np.linalg.norm(self.y_colmap)

        
        self.x_real = self.cfg['x_real']
        self.y_real = self.cfg['y_real']

        self.scale_factor = (self.x_real/self.x_colmap_norm + self.y_real/self.y_colmap_norm)/2
        self.compute_goal_plane()
        self.compute_obj = ComputeRotation(self.origin_colmap,self.x_colmap,self.y_colmap,self.goal_plane)
        self.transform = self.compute_obj.transform_plane()
        
    def load_poses(self):
        print('loading poses')
        with open(self.args.json_in) as f:
            self.poses = json.load(f)
        self.json_out = self.poses.copy()
        self.frames_in = self.poses['frames']
        self.frames_out = self.json_out['frames']
        self.n_frames = len(self.frames_in)
        print('n_frames: ',self.n_frames)

    def compute_goal_plane(self):
        # cross product of x and y
        self.goal_plane = np.cross(self.x_colmap/self.x_colmap_norm,self.y_colmap/self.y_colmap_norm)
        d = -np.dot(self.goal_plane,self.origin_colmap)
        # d = 0
        self.goal_plane = np.append(self.goal_plane,d)
        

    def visualize_plane(self,ax,plane,plot_vectors=True):
        # visualize plane
        a,b,c,d = plane
        x = np.linspace(-10,10,100)
        y = np.linspace(-10,10,100)
        X,Y = np.meshgrid(x,y)
        Z = (-a*X - b*Y - d)/c
        # fig = plt.figure()
        # ax = plt.axes(projection='3d')
        ax.plot_surface(X,Y,Z,alpha=0.2)

        # visualize vectors x_colmap and y_colmap starting at origin_colmap
        if plot_vectors:
            ax.quiver(self.origin_colmap[0],self.origin_colmap[1],self.origin_colmap[2],self.x_colmap[0],self.x_colmap[1],self.x_colmap[2],color='r')
            ax.quiver(self.origin_colmap[0],self.origin_colmap[1],self.origin_colmap[2],self.y_colmap[0],self.y_colmap[1],self.y_colmap[2],color='g')
        else:
            # transform vectors using self.transform
            x_colmap_transformed = np.dot(self.transform,np.append(self.x_colmap,1))
            y_colmap_transformed = np.dot(self.transform,np.append(self.y_colmap,1))
            origin_transformed = np.dot(self.transform,np.append(self.origin_colmap,1))
            ax.quiver(origin_transformed[0],origin_transformed[1],origin_transformed[2],x_colmap_transformed[0],x_colmap_transformed[1],x_colmap_transformed[2],color='r')
            ax.quiver(origin_transformed[0],origin_transformed[1],origin_transformed[2],y_colmap_transformed[0],y_colmap_transformed[1],y_colmap_transformed[2],color='g')

        plt.show()
    
    def visualize_new_poses(self):
        plt.clf()
        args.bound = 1.0
        visualizer = CameraPoseVisualizer([-args.bound, args.bound], [-args.bound, args.bound], [-args.bound, args.bound])
        for i in range(self.n_frames):
            P = np.array(self.frames_out[i]['transform_matrix'])
            type = self.frames_out[i]['type']
            if type == 'wrap':
                visualizer.extrinsic2pyramid(P, 'c', 0.05)
            elif type == 'bg':
                visualizer.extrinsic2pyramid(P, 'g', 0.05)
        
        plt.show()
        # self.visualize_plane(visualizer.ax,plane=self.compute_obj.new_plane,plot_vectors=False)

    def process(self):
        for i in range(self.n_frames):
            P = np.array(self.frames_in[i]['transform_matrix'])
                
            type = self.frames_in[i]['type']
            if type == 'wrap':
                visualizer.extrinsic2pyramid(P, 'c', 0.25)
            elif type == 'bg':
                visualizer.extrinsic2pyramid(P, 'g', 0.25)
            # now rotate s.t. x_colmap aligns with x_real         
            P = np.dot(self.transform,P)
        
            # now scale 
            P[:3,3] = P[:3,3] * self.scale_factor

            self.frames_out[i]['transform_matrix'] = P.tolist()
        
        self.visualize_plane(visualizer.ax,plane=self.goal_plane)
        
        self.json_out['frames'] = self.frames_out
        
    def save_json(self):
        with open(self.args.json_out,'w') as f:
            json.dump(self.json_out,f,indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process colmap poses for easy robot execution.')
    parser.add_argument('--json_in', type=str, help='Path to colmap poses json file.')
    parser.add_argument('--json_out', type=str, help='Path to output json file.',default='poses_robot.json')
    parser.add_argument('--cfg' , type=str, help='Path to config file.')
    parser.add_argument('--bound',type=float,default=5.0)
    args = parser.parse_args()

    visualizer = CameraPoseVisualizer([-args.bound, args.bound], [-args.bound, args.bound], [-args.bound, args.bound])
    
    processor = Processor(args)