import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.cm as cm

# Read rotation data_base
file_path_rot = 'rotation_joints.trc'
rot_data = pd.read_csv(file_path_rot, delimiter='\t', skiprows=3)

# Read positions data_base
file_path_pos = 'position_joints.mot'
pos_data = pd.read_csv(file_path_pos, delimiter='\t', skiprows=11)



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