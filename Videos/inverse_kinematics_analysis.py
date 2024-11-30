import numpy as np
import matplotlib.pyplot as plt
import torch
import joblib

# Load the inverse kinematics points from the file
file_path = 'valse_constantin.pt'
points = joblib.load(file_path)

keys = list(points.keys())

print('keys', keys)

#keys ['pred_uvd', 'pred_xyz_29', 'pred_scores', 'pred_sigma', 'f', 'pred_betas', 'pred_phi', 'scale_mult', 'pred_cam_root', 'bbox', 'height', 'width', 'img_path', 'img_sizes']
print('camera root', points['pred_cam_root']) #Position of the camera
print('pred_uvd', points['pred_uvd']) #Predicted 3D coordinates of the different joints
print('pred_xyz', points['pred_xyz_29']) #Prediction of the 3D coordinates of the different joints
print('pred_scores', points['pred_scores']) # Scores of the different joints
print('pred_sigma', points['pred_sigma']) #Sigma of the camera  (uncertainty of the camera)
print('f', points['f']) #Focal length of the camera
print('pred_betas', points['pred_betas']) #Body shape parameters of the person in the image 
print('pred_phi', points['pred_phi']) #Phi angle of the camera
print('scale_mult', points['scale_mult']) #Vide
print('bbox', points['bbox']) #Bounding box of the image