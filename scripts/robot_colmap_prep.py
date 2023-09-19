import numpy as np
import json 
import os  
import argparse



class Processor:
    def __init__(self,args):
        self.args = args
        self.load_config()
        self.load_poses()
        self.process()
        # self.save_json()

    def load_config(self):
        with open(self.args.cfg) as f:
            self.cfg = json.load(f)
        self.x_colmap = np.array(self.cfg['x'])
        self.y_colmap = np.array(self.cfg['y'])
        self.x_colmap_norm = np.linalg.norm(self.x_colmap)
        self.y_colmap_norm = np.linalg.norm(self.y_colmap)

        self.origin_colmap = np.array(self.cfg['origin'])
        self.x_real = self.cfg['x_real']
        self.y_real = self.cfg['y_real']

        self.scale_factor = (self.x_real/self.x_colmap_norm + self.y_real/self.y_colmap_norm)/2
    
    def load_poses(self):
        print('loading poses')
        with open(self.args.json_in) as f:
            self.poses = json.load(f)
        self.json_out = self.poses.copy()
        self.frames_in = self.poses['frames']
        self.frames_out = self.json_out['frames']
        self.n_frames = len(self.frames_in)

    def process(self):
        for i in range(self.n_frames):
            # first align with origin 
            P = np.array(self.frames_in[i]['transform_matrix'])
            P[:3,3] = P[:3,3] - self.origin_colmap

            # now rotate s.t. x_colmap aligns with x_real
            # TODO

            
            # now scale 
            P[:3,3] = P[:3,3] * self.scale_factor

            

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process colmap poses for easy robot execution.')
    parser.add_argument('--json_in', type=str, help='Path to colmap poses json file.')
    parser.add_argument('--json_out', type=str, help='Path to output json file.')
    parser.add_argument('--cfg' , type=str, help='Path to config file.')
    args = parser.parse_args()
    processor = Processor(args)
    
