#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: Apache-2.0

#TODO CVT: - Jouer avec le poids des joints pour voir si on peut améliorer le mouvement
# - Faire en sorte que le robot puisse continuer à danser sans s'arrêter. Faire boucler le mouvement ?
# - Plot le mouvement le mouvement du robot en 2D (x, y) sur matplotlib
# - Pour cela voir à quelle frame il revient sur son point de départ
# - Corriger la reconnaissance du rythme - Jouer le time.sleep sur chaque frame.
# - Générer la vidéo.


"""DRACO 3 humanoid standing on two feet and reaching with a hand."""

import joblib
import numpy as np
import pandas as pd
import pinocchio as pin
import qpsolvers
import time
from scipy.signal import savgol_filter

import meshcat_shapes
import pink
from pink import solve_ik
from pink.tasks import FrameTask, JointCouplingTask, PostureTask
from pink.visualization import start_meshcat_visualizer

try:
    from robot_descriptions.loaders.pinocchio import load_robot_description
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "Examples need robot_descriptions, "
        "try `pip install robot_descriptions`"
    ) from exc

results = joblib.load('valse_constantin.pt')
positions = results['pred_xyz_29']
perfect_positions = False
init_frame = 0.
max_frame = positions.shape[0] - 1

element_markers = {"pelvis": 0, "r_hand": 26, "l_hand": 25, "head": 24, "r_foot": 28, "l_foot": 27}
element_costs = {"pelvis": [1., 0.], "r_hand": [.7, 0.], "l_hand": [.7, 0.], "head": [1., 0.], "r_foot": [1., 0.], "l_foot": [1., 0.]}
# It is more natural to get rotations to 0 in order to avoid noise.


# Getting the optimal transformation matrix to align the movement with the (x, y) plane
head_pose = positions[int(init_frame):, element_markers["head"]]
pelvis_pose = positions[int(init_frame):, element_markers["pelvis"]]
r_foot_pose = positions[int(init_frame):, element_markers["r_foot"]]
l_foot_pose = positions[int(init_frame):, element_markers["l_foot"]]

def get_transformation_matrix(values):
    """Get the transformation matrix from the parameters.
    
    Args:
        values: List of 9 values [alpha, beta, gamma, x, y, z, sx, sy, sz]
        alpha, beta, gamma: Rotation angles around x, y, z axes
        x, y, z: Translation along x, y, z axes
        sx, sy, sz: Scaling along x, y, z axes
        
    Returns:
        trans_mat: Transformation matrix
    """
    rot_angles = values[:3]; translation = values[3:6]; scale = values[6:9]
    alpha = rot_angles[0]; beta = rot_angles[1]; gamma = rot_angles[2]
    ca = np.cos(alpha); sa = np.sin(alpha)
    cb = np.cos(beta); sb = np.sin(beta)
    cg = np.cos(gamma); sg = np.sin(gamma)
    rot_mat = np.array([[ca*cb, ca*sb*sg - sa*cg, ca*sb*cg + sa*sg],
                        [sa*cb, sa*sb*sg + ca*cg, sa*sb*cg - ca*sg],
                        [-sb, cb*sg, cb*cg]])
    rot_mat[:3, 0] *= scale[0]
    rot_mat[:3, 1] *= scale[1]
    rot_mat[:3, 2] *= scale[2]
    trans_mat = np.eye(4)
    trans_mat[:3, :3] = rot_mat
    trans_mat[:3, 3] = translation
    return trans_mat

def apply_transformation(transformation_matrix, positions):
    """Apply transformation matrix to the positions.
    
    Args:
        transformation_matrix: Transformation matrix, shape (4, 4)
        positions: Positions to be transformed, shape (t, n, 3)
    """
    if positions.ndim == 2:
        positions_bigger = np.concatenate((positions, np.ones((positions.shape[0], 1))), axis=1)
        transformed_positions_bigger = (transformation_matrix@positions_bigger.T).T
        transformed_positions = transformed_positions_bigger[:, :3]
        
    else:
        transformed_positions = np.zeros_like(positions)
        for t in range(positions.shape[0]):
            actual_position = positions[t]
            actual_position_bigger = np.concatenate((actual_position, np.ones((actual_position.shape[0], 1))), axis=1)
            transformed_position_bigger = (transformation_matrix@actual_position_bigger.T).T
            transformed_positions[t] = transformed_position_bigger[:, :3]
            
    return transformed_positions

def cost(rot_angles):
    trans_mat = get_transformation_matrix(rot_angles)
    transformed_head = apply_transformation(transformation_matrix=trans_mat, positions=head_pose)
    transformed_pelvis = apply_transformation(transformation_matrix=trans_mat, positions=pelvis_pose)
    transformed_r_foot = apply_transformation(transformation_matrix=trans_mat, positions=r_foot_pose)
    transformed_l_foot = apply_transformation(transformation_matrix=trans_mat, positions=l_foot_pose)

    cost = np.sum(transformed_head[:, 2] - 1.5)**2
    cost += np.sum(transformed_pelvis[:, 0] - transformed_head[:, 0])**2
    cost += np.sum(transformed_pelvis[:, 1] - transformed_head[:, 1])**2
    cost += np.abs(np.sum(transformed_l_foot[:, 2]))
    cost += np.abs(np.sum(transformed_r_foot[:, 2]))
    # cost = np.var(transformed_pelvis[:, 2])
    # cost += np.mean(transformed_feet[:, 1] - transformed_head[:, 1])**2
    return cost

def get_positions_of_base_frame(niki_positions, perfect_positions=False):
    """Get the positions of the base frame in the world frame.
    
    Args:
        niki_positions: Positions of the robot in the world frame, shape (n, 3)
        perfect_positions: Boolean, whether the positions are from Niki or perfect positions
        
    Returns:
        base_frame_positions: Positions of the base frame in the world frame, shape (n, 3)
    """
    if perfect_positions:
        # Paramètres de la musique
        BPM = 187
        small_circle_duration = 6*60/BPM
        print(f"Durée d'un petit cercle : {small_circle_duration:.2f} secondes")

        # Paramètres pour le mouvement circulaire (petit cercle)
        r_circle = 0.5 # rayon du petit cercle (en mètres)
        omega_circle = -2 * np.pi / small_circle_duration  # vitesse angulaire (rad/s)

        # Paramètres pour le mouvement elliptique (grand cercle de bal)
        a_ellipse = 5  # demi-grand axe de l'ellipse (en mètres)
        b_ellipse = 3  # demi-petit axe de l'ellipse (en mètres)
        omega_ellipse = 2 * np.pi / 60  # vitesse pseudo-angulaire (rad/s)

        # Temps
        number_frames = positions.shape[0]
        number_seconds = number_frames/30
        t = np.linspace(0, number_seconds, number_frames)  # 60 secondes, 1000 points

        # Mouvement circulaire
        x_circle = r_circle * np.cos(omega_circle * t)
        y_circle = r_circle * np.sin(omega_circle * t)

        # Mouvement elliptique
        x_ellipse = a_ellipse * np.cos(omega_ellipse * t)
        y_ellipse = b_ellipse * np.sin(omega_ellipse * t)

        # Composition des mouvements
        x_total = x_circle + x_ellipse
        y_total = y_circle + y_ellipse
        z_total = np.zeros_like(x_total)
    
    else:
        x_total = niki_positions[:, 0]
        y_total = niki_positions[:, 1]
        x_total = savgol_filter(x_total, 100, 3)
        y_total = savgol_filter(y_total, 100, 3)
        z_total = np.zeros_like(x_total)

    base_frame_positions = np.concatenate((x_total.reshape(-1, 1), y_total.reshape(-1, 1), z_total.reshape(-1, 1)), axis=1)
    return base_frame_positions

values_0 = [np.pi, 0., -np.pi/2, 0., 0., 1., 2., 2., 2.]
# values_opt = fmin_bfgs(cost, values_0)
trans_mat = get_transformation_matrix(values_0)

# Apply transformation matrix to all the positions
transformed_positions = apply_transformation(transformation_matrix=trans_mat, positions=positions)

nike_base_positions_orig = results['pred_cam_root']
transformed_nike_base_positions = apply_transformation(transformation_matrix=trans_mat, positions=nike_base_positions_orig)
base_frame_positions = get_positions_of_base_frame(niki_positions=transformed_nike_base_positions - transformed_nike_base_positions[0])

for t in range(transformed_positions.shape[0]):
    transformed_positions[t] += base_frame_positions[t]

element_positions = {element: transformed_positions[:, element_markers[element]] for element in element_markers}

class RobotElementPose:
    """ Base class used to define the position of a robot element at a given time."""
    def __init__(self, init_time: float, configuration: pink.Configuration, element_name: str):
        """Initialize pose.

        Args:
            init: Initial transform from the element frame to the world frame.
        """
        self.init_time = init_time
        self.element_name = element_name
        init_config = configuration.get_transform_frame_to_world(element_name)
        init_position = element_positions[element_name][int(init_time)]
        self.init = init_config.copy()
        self.init.translation = init_position

    def at(self, t):
        """Get element pose at a given time.

        Args:
            t: Time in seconds.
        """
        T = self.init.copy()
        position = element_positions[self.element_name][int(t)]
        # R = T.rotation
        # R = np.dot(R, pin.utils.rpyToMatrix(0., 0., np.pi / 2))
        # R = np.dot(R, pin.utils.rpyToMatrix(0., -np.pi, 0.))
        # T.rotation = R
        T.translation = position
        return T

if __name__ == "__main__":
    robot = load_robot_description(
        "atlas_v4_description", root_joint=pin.JointModelFreeFlyer()
    )

    # Initialize visualization
    viz = start_meshcat_visualizer(robot)
    element_frames = {}
    for element in element_markers:
        element_frames[element] = viz.viewer[element + "_pose"]
        meshcat_shapes.frame(element_frames[element])

    # Set initial robot configuration
    configuration = pink.Configuration(robot.model, robot.data, robot.q0)
    viz.display(configuration.q)

    # Tasks initialization for IK
    element_tasks = {}
    for element in element_markers:
        element_tasks[element] = FrameTask(
            element,
            position_cost=element_costs[element][0],
            orientation_cost=element_costs[element][1],
        )
        
    posture_task = PostureTask(
        cost=1.e-1,  # [cost] / [rad]
    )

    tasks = [element_tasks[element] for element in element_markers]
    tasks += [
        posture_task,
    ]

    # Task target specifications
    for element in element_markers:
        element_tasks[element].set_target(
            configuration.get_transform_frame_to_world(element)
        )
    posture_task.set_target_from_configuration(configuration)

    # Initialize robot element poses
    element_poses = {}
    for element in element_markers:
        element_poses[element] = RobotElementPose(init_frame, configuration, element)

    # Select QP solver
    solver = qpsolvers.available_solvers[0]
    if "quadprog" in qpsolvers.available_solvers:
        solver = "quadprog"

    # rate = RateLimiter(frequency=200., warn=False)
    # dt = rate.period
    dt = 1.
    t = init_frame  # [s]
    while True:
        # Update task targets
        for element in element_markers:
            element_tasks[element].set_target(element_poses[element].at(t))
            element_frames[element].set_transform(element_poses[element].at(t).np)

        # Compute velocity and integrate it into next configuration
        velocity = solve_ik(configuration, tasks, dt, solver=solver)
        configuration.integrate_inplace(velocity, dt)

        # Visualize result at fixed FPS
        viz.display(configuration.q)
        t += dt
        if t > max_frame:
            print("Restarting animation")
            t = init_frame
        time.sleep(0.03)