import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.cm as cm


def plot_position_data(time_lim_inf=None, time_lim_sup=None):
    
    # Read position data
    file_path_pos = 'position_joints.trc'
    data_pos = pd.read_csv(file_path_pos, delimiter='\t', skiprows=3)

    # Extract time and marker positions
    time_pos = data_pos['Time'] # Time column
    num_markers = 22  # From the file header description, there are 22 markers

    # Extract X, Y, Z coordinates for each marker
    marker_columns = [f'X{i} Y{i} Z{i}'.split() for i in range(1, num_markers + 1)]
    flat_marker_columns = [col for sublist in marker_columns for col in sublist]

    # Ensure the columns exist in the data
    missing_columns = [col for col in flat_marker_columns if col not in data_pos.columns]
    if missing_columns:
        raise KeyError(f"Missing columns in the data: {missing_columns}")

    # Extract marker data
    marker_data = {f'Marker {i}': data_pos[columns] for i, columns in enumerate(marker_columns, start=1)}

    # Set up a colormap for unique colors for each marker
    cmap = cm.get_cmap('tab20', num_markers)
    colors = [cmap(i) for i in range(num_markers)]

    # Plot 3D dynamic visualization of all markers
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Select a subset of time points to plot
    if time_lim_inf is not None and time_lim_sup is not None:
        time_selected = time_pos[(time_pos >= time_lim_inf) & (time_pos <= time_lim_sup)]
    else:
        time_selected = time_pos

    for t in time_selected.index:
        ax.cla()
        for i, (marker, coords) in enumerate(marker_data.items()):
            x, y, z = coords.iloc[t]
            ax.scatter(x, y, z, c=[colors[i]], label=marker if t == time_selected.index[0] else "", marker='o')  # Add legend only on first frame
    
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Time: {time_pos.iloc[t]:.2f}')
        if t == time_selected.index[0]:
            ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1))  # Add legend only once
        plt.pause(0.0344828)  # Pause to create an animation effect

    plt.show()

    # Add a time series plot the X markers coordinates over time
    fig, ax = plt.subplots()
    for i, (marker, coords) in enumerate(marker_data.items()):
        ax.plot(time_pos, coords.iloc[:, 0], label=f'{marker} X', color=colors[i])  # Example: Plot X coordinate over time
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Coordinate Value')
    ax.set_title('Marker X Coordinate Over Time')
    ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
    plt.show()



def read_data_rot():
    # Read rotation data
    file_path_rot = 'rotation_joints.mot'
    data_rot = pd.read_csv(file_path_rot, delimiter='\t', skiprows=10)

    # Extract time column
    time_rot = data_rot['time']

    # Identify joints with their 3D positions
    joint_names = [
        'pelvis_tilt', 'pelvis_list', 'pelvis_rotation', 'pelvis_tx', 'pelvis_ty', 'pelvis_tz',
        'hip_flexion_r', 'hip_adduction_r', 'hip_rotation_r', 'knee_angle_r', 'knee_angle_r_beta',
        'ankle_angle_r', 'subtalar_angle_r', 'mtp_angle_r', 'hip_flexion_l', 'hip_adduction_l',
        'hip_rotation_l', 'knee_angle_l', 'knee_angle_l_beta', 'ankle_angle_l', 'subtalar_angle_l',
        'mtp_angle_l', 'L5_S1_Flex_Ext', 'L5_S1_Lat_Bending', 'L5_S1_axial_rotation', 'L4_L5_Flex_Ext',
        'L4_L5_Lat_Bending', 'L4_L5_axial_rotation', 'L3_L4_Flex_Ext', 'L3_L4_Lat_Bending',
        'L3_L4_axial_rotation', 'L2_L3_Flex_Ext', 'L2_L3_Lat_Bending', 'L2_L3_axial_rotation',
        'L1_L2_Flex_Ext', 'L1_L2_Lat_Bending', 'L1_L2_axial_rotation', 'L1_T12_Flex_Ext',
        'L1_T12_Lat_Bending', 'L1_T12_axial_rotation', 'Abs_r3', 'Abs_r2', 'Abs_r1', 'Abs_t1', 'Abs_t2',
        'neck_flexion', 'neck_bending', 'neck_rotation', 'arm_flex_r', 'arm_add_r', 'arm_rot_r',
        'elbow_flex_r', 'pro_sup_r', 'wrist_flex_r', 'wrist_dev_r', 'arm_flex_l', 'arm_add_l',
        'arm_rot_l', 'elbow_flex_l', 'pro_sup_l', 'wrist_flex_l', 'wrist_dev_l'
    ]

    joints_3d = {joint: data_rot[[f'{joint}_tx', f'{joint}_ty', f'{joint}_tz']] for joint in joint_names}

    # Set up a colormap for unique colors for each joint
    cmap = cm.get_cmap('tab20', len(joint_names))
    colors = [cmap(i) for i in range(len(joint_names))]

    # Dynamically plot the 3D movement of all joints over time
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for t in range(len(time_rot)):
        ax.clear()
        for i, (joint, coords) in enumerate(joints_3d.items()):
            x, y, z = coords.iloc[t]
            ax.scatter(x, y, z, c=[colors[i]], label=joint if t == 0 else "", marker='o')  # Add legend only once

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Time: {time_rot[t]:.2f}')
        if t == 0:
            ax.legend()  # Add legend only for the first frame
        plt.pause(0.1)  # Pause to create an animation effect

    plt.show()

    # Plot angular data (e.g., pelvis_tilt) for all joints over time
    fig, ax = plt.subplots()

    for i, joint in enumerate(joint_names):
        ax.plot(time_rot, data_rot[f'{joint}_tilt'], label=joint, color=colors[i])  # Example: 'joint_tilt'

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Angle (degrees)')
    ax.set_title('Joint Angles over Time')
    ax.legend()
    plt.show()

# Call the function to read and plot rotation data
#read_data_rot()

# Call the function to read and plot position data
plot_position_data(time_lim_inf=8)

# Conclusion: Will use the video from 8.00000 seconds
