'''This file plot the 3D coordinates of the different joints dynamically over time and the variations of phi over time
The file is used to analyze the inverse kinematics points obtained from the file valse_constantin.pt
Those are the simulations calculted with niki. The file is located in Research/obtain_kinematics_niki/inverse_kinematics_analysis.py'''

import matplotlib.pyplot as plt
import joblib
import matplotlib.cm as cm

# Load the inverse kinematics points from the file
file_path = 'valse_constantin.pt'
points = joblib.load(file_path)

keys = list(points.keys())

print('keys', keys)

#keys ['pred_uvd', 'pred_xyz_29', 'pred_scores', 'pred_sigma', 'f', 'pred_betas', 'pred_phi', 'scale_mult', 'pred_cam_root', 'bbox', 'height', 'width', 'img_path', 'img_sizes']
print('camera root', points['pred_cam_root']) #Position of the camera
print('pred_uvd', points['pred_uvd']) #Predicted 3D coordinates of the different joints
print('pred_uvd_forme', points['pred_uvd'].shape) #Shape of the predicted 3D coordinates of the different joints
print('pred_xyz', points['pred_xyz_29']) #Prediction of the 3D coordinates of the different joints
print('pred_xyz_forme', points['pred_xyz_29'].shape) #Shape of the prediction of the 3D coordinates of the different joints
print('pred_scores', points['pred_scores']) # Scores of the different joints
print('pred_scores_forme', points['pred_scores'].shape) #Shape of the scores of the different joints
print('pred_sigma', points['pred_sigma']) #Sigma of the different joints
print('pred_sigma_forme', points['pred_sigma'].shape) #Shape of the sigma of the camera
print('f', points['f']) #Focal length of the camera
print('pred_betas', points['pred_betas']) #10 main components of the body shape of the person in the image
print('pred_betas_forme', points['pred_betas'].shape) 
print('pred_phi', points['pred_phi']) #Phi angle of 23 joints of the camera
print('pred_phi_forme', points['pred_phi'].shape) #Shape of the phi angle of the camera
print('scale_mult', points['scale_mult']) #Vide
print('bbox', points['bbox']) #Bounding box of the image

# Plot the 3D coordinates of the different joints dynamically over time
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Assuming points['pred_xyz_29'] is of shape (T, 29, 3) where T is the number of time steps
num_time_steps = points['pred_xyz_29'].shape[0]

for t in range(num_time_steps):
    ax.clear()
    ax.scatter(points['pred_xyz_29'][t, :, 0], points['pred_xyz_29'][t, :, 1], points['pred_xyz_29'][t, :, 2])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Time step {t}')
    #plt.pause(0.001)  # Pause to create an animation effect
plt.show()

# PLot the variations of phi over time
fig, ax = plt.subplots()

# Create a color map
cmap = cm.get_cmap('viridis', points['pred_phi'].shape[1])

for i in range(points['pred_phi'].shape[1]):
    ax.plot(points['pred_phi'][:, i], label=f'Joint {i}', color=cmap(i))
ax.set_xlabel('Time')
ax.set_ylabel('Phi')
ax.set_title('Variation of Phi over time')
plt.legend()
plt.show()