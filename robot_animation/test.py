import numpy as np
import pink
from pink.tasks import FrameTask, PostureTask
from pink import solve_ik
from robot_descriptions.loaders.pinocchio import load_robot_description
import pandas as pd
import time 
import meshcat
import meshcat.geometry as g
from pinocchio.visualize import MeshcatVisualizer
from pink.visualization import start_meshcat_visualizer

# Initialize MeshCat visualizer
vis = meshcat.Visualizer().open()

# Load the robot description
robot = load_robot_description("atlas_v4_description")
configuration = pink.Configuration(robot.model, robot.data, robot.q0)

# Add robot model to the visualizer
visualizer = MeshcatVisualizer(robot.model, robot.collision_model, robot.visual_model)
visualizer.initViewer(viewer=vis)
visualizer.loadViewerModel()

# Define tasks for Pink
tasks = {
    "pelvis": FrameTask("pelvis", position_cost=1.0, orientation_cost=1.0),
    "l_foot": FrameTask("l_foot", position_cost=1.0, orientation_cost=1.0),
    "r_foot": FrameTask("r_foot", position_cost=1.0, orientation_cost=1.0),
    "posture": PostureTask(cost=1e-3),
}

# Set posture task to the robot's default configuration
tasks["posture"].set_target(configuration.q)

# Read position data
file_path_pos = 'position_joints.trc'
data_pos = pd.read_csv(file_path_pos, delimiter='\t', skiprows=2)

# Extract time and marker positions
if 'Time' not in data_pos.columns:
    raise KeyError("The 'Time' column is missing from the position data file.")
time_col = pd.to_numeric(data_pos['Time'], errors='coerce')

# Map robot joints to TRC markers (update with actual mappings)
joint_marker_map = {
    "back_bkz": "X1 Y1 Z1".split(),  # Spine base
    "l_arm_shz": "X5 Y5 Z5".split(),  # Left shoulder
    "r_arm_shz": "X13 Y13 Z13".split(),  # Right shoulder
    "l_leg_hpz": "X10 Y10 Z10".split(),  # Left hip
    "r_leg_hpz": "X4 Y4 Z4".split(),  # Right hip
    "neck_ry": "X14 Y14 Z14".split(),  # Neck
}

# Find index for 8 seconds
time_8_sec_idx = np.argmin(np.abs(time_col - 8.0))

# Extract positions at 8 seconds
initial_positions = {}
for joint, markers in joint_marker_map.items():
    if all(marker in data_pos.columns for marker in markers):
        initial_positions[joint] = data_pos.loc[time_8_sec_idx, markers].values
        print(f"Initial position for {joint}: {initial_positions[joint]}")

# Set the initial configuration for the tasks
for joint, position in initial_positions.items():
    if joint in tasks:
        tasks[joint].set_target(pink.geometry.SE3(position, np.eye(3)))

# Visualize the initial configuration
viz = start_meshcat_visualizer(robot)
visualizer.display(configuration.q)

# Animation loop (same as before)
dt = 0.034  # Time step from the data (29 Hz frame rate)
feet_markers = {"l_leg_mtp": ["X11", "Y11", "Z11"], "r_leg_mtp": ["X5", "Y5", "Z5"]}

for idx in range(len(time_col)):
    # Update feet positions
    for foot, markers in feet_markers.items():
        position = data_pos.loc[idx, markers].values
        if foot in tasks:
            tasks[foot].set_target(pink.geometry.SE3(position, np.eye(3)))

    # Solve IK and update configuration
    velocity = solve_ik(configuration, tasks.values(), dt, solver="quadprog")
    configuration.integrate_inplace(velocity, dt)

    # Optional: Visualize or log configuration
    print(f"Frame {idx}, Configuration: {configuration.q}")

    time.sleep(dt)  # Simulate real-time animation speed