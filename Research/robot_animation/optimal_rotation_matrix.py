'''Find optimal rotation matrix to reduce to a minimum variance of coordinates on the z-axis.'''

import pandas as pd
import numpy as np
from scipy.optimize import fmin_bfgs

data_pos = pd.read_csv('position_joints.trc', delimiter='\t', skiprows=3)
head_pose = np.array(data_pos.loc[275:, ["X15", "Y15", "Z15"]])
feet_pose = np.array(data_pos.loc[275:, ["X5", "Y5", "Z5"]])

head_pose.shape

start_head_pose = head_pose[0]

def cost(rot_coefs):
    # The cost is the absolute mean z-coordinate of the head
    rot_mat = np.array([rot_coefs[0:3], rot_coefs[3:6], rot_coefs[6:9]])
    rotated_head = head_pose@rot_mat
    rotated_feet = feet_pose@rot_mat
    cost = np.max(rotated_head[:, 2] - 1.5)**2
    cost += np.max(rotated_feet[:, 2])**2
    return cost

rot_0 = [1, 0, 0, 0, 1, 0, 0, 0, 1]
rot_opt = fmin_bfgs(cost, rot_0)

print(rot_opt)
print(cost(rot_opt))
print(np.array([rot_opt[0:3], rot_opt[3:6], rot_opt[6:9]]))

print(head_pose@(np.array([rot_opt[0:3], rot_opt[3:6], rot_opt[6:9]])))