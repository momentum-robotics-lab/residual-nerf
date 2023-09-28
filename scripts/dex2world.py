import os
import argparse 
import json 
import numpy as np 

parser = argparse.ArgumentParser(description='Convert dex to world')
parser.add_argument('--depth',type=float,required=True)
parser.add_argument('--grasp_x',type=int,required=True)
parser.add_argument('--grasp_y',type=int,required=True)
parser.add_argument('--fov_x',type=int,default=50,help='veritcal fov in degrees')
parser.add_argument('--res_x',type=int,default=270)
parser.add_argument('--res_y',type=int,default=480)
parser.add_argument('--cropped_res',default=200,type=int)
args = parser.parse_args()

# compute K 
f = args.res_x / (2 * np.tan(args.fov_x / 2 * np.pi / 180))
c_x = args.res_x / 2
c_y = args.res_y / 2
K = np.array([[f, 0, c_x], [0, f, c_y], [0, 0, 1]])
K_inv= np.array([[1.0/f, 0      , -c_x / f],
                [ 0,      1.0/f, -c_y / f],
                [ 0,      0      ,                1.0]])



crop_start_coord = np.array([args.res_x//2-args.cropped_res//2,args.res_y//2-args.cropped_res//2])
grasp_location = np.array([args.grasp_y,args.grasp_x]) + crop_start_coord # my coordinate system means x is horizontal

print('global grasp location w.r.t. center',grasp_location)

grasp_camframe = K_inv @ np.array([grasp_location[0],grasp_location[1],1])
grasp_camframe *= args.depth/np.linalg.norm(grasp_camframe)

goal_position = np.array([0.17, 0.73025*0.5,0.80])

global_grasp = np.array([goal_position[0]+grasp_camframe[0],goal_position[1]+grasp_camframe[1],goal_position[2]-grasp_camframe[2]]) 
print('global_grasp',global_grasp)

