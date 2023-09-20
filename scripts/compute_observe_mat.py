import numpy as np
from util.camera_pose_visualizer import CameraPoseVisualizer

def nerf_matrix_to_ngp(pose, scale=0.33, offset=[0, 0, 0]):
    # for the fox dataset, 0.33 scales camera radius to ~ 2
    new_pose = np.array([
        [pose[1, 0], -pose[1, 1], -pose[1, 2], pose[1, 3] * scale + offset[0]],
        [pose[2, 0], -pose[2, 1], -pose[2, 2], pose[2, 3] * scale + offset[1]],
        [pose[0, 0], -pose[0, 1], -pose[0, 2], pose[0, 3] * scale + offset[2]],
        [0, 0, 0, 1],
    ], dtype=np.float32)

    return new_pose


def R_z(theta):
    return np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1],
    ], dtype=np.float32)


visualizer = CameraPoseVisualizer([-1, 1], [-1, 1], [-0, 1])
scale = 2.0 # ngp environment w.r.t. real
goal_position = np.array([0.20, 0.73025*0.5,0.80]) # these numbers (except scale) are what we want in the real world

P = np.eye(4)
P[0, 3] = goal_position[0]
P[1, 3] = goal_position[1]
P[2, 3] = goal_position[2]
P[:3, :3] = R_z(np.pi/2)
print(P)
print(nerf_matrix_to_ngp(P,scale=scale))

# hardcoded_mat = np.load('pose.npy')
# print(hardcoded_mat)

np.save('pose.npy', nerf_matrix_to_ngp(P,scale=scale))

visualizer.extrinsic2pyramid(P, 'c', 0.25)
visualizer.show()





