#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 Inria

"""DRACO 3 humanoid standing on two feet and reaching with a hand."""

import joblib
import numpy as np
import pandas as pd
import pinocchio as pin
import qpsolvers
import time
from scipy.optimize import fmin_bfgs
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
adding_perfect_positions = False

feet_markers = {"l_foot": 28, "r_foot": 27}
pelvis_markers = {"pelvis": 0}
head_markers = {"head": 24}
init_frame = 0.0
head_pose = positions[int(init_frame):, head_markers["head"]]
pelvis_pose = positions[int(init_frame):, pelvis_markers["pelvis"]]
r_foot_pose = positions[int(init_frame):, feet_markers["r_foot"]]
l_foot_pose = positions[int(init_frame):, feet_markers["l_foot"]]

def get_transformation_matrix(values):
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

def cost(rot_angles):
    trans_mat = get_transformation_matrix(rot_angles)
    head_pose_bigger = np.concatenate((head_pose, np.ones((head_pose.shape[0], 1))), axis=1)
    pelvis_pose_bigger = np.concatenate((pelvis_pose, np.ones((pelvis_pose.shape[0], 1))), axis=1)
    r_foot_pose_bigger = np.concatenate((r_foot_pose, np.ones((r_foot_pose.shape[0], 1))), axis=1)
    l_foot_pose_bigger = np.concatenate((l_foot_pose, np.ones((l_foot_pose.shape[0], 1))), axis=1)

    rotated_head_bigger = trans_mat@head_pose_bigger.T
    rotated_pelvis_bigger = trans_mat@pelvis_pose_bigger.T
    rotated_r_foot_bigger = trans_mat@r_foot_pose_bigger.T
    rotated_l_foot_bigger = trans_mat@l_foot_pose_bigger.T

    rotated_head = rotated_head_bigger[:3].T
    rotated_pelvis = rotated_pelvis_bigger[:3].T
    rotated_r_foot = rotated_r_foot_bigger[:3].T
    rotated_l_foot = rotated_l_foot_bigger[:3].T

    cost = np.sum(rotated_head[:, 2] - 1.5)**2
    cost += np.sum(rotated_pelvis[:, 0] - rotated_head[:, 0])**2
    cost += np.sum(rotated_pelvis[:, 1] - rotated_head[:, 1])**2
    cost += np.abs(np.sum(rotated_l_foot[:, 2]))
    cost += np.abs(np.sum(rotated_r_foot[:, 2]))
    # cost = np.var(rotated_pelvis[:, 2])
    # cost += np.mean(rotated_feet[:, 1] - rotated_head[:, 1])**2
    return cost

def get_positions_of_base_frame(camera_positions):
    if adding_perfect_positions:
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
        x_total = camera_positions[:, 0]
        y_total = camera_positions[:, 1]
        x_total = savgol_filter(x_total, 100, 3)
        y_total = savgol_filter(y_total, 100, 3)
        z_total = np.zeros_like(x_total)

    return x_total, y_total, z_total

values_0 = [0.0, 0.0, -np.pi/2, 0.0, 0.0, 1.0, 2.0, 2.0, 2.0]
# values_opt = fmin_bfgs(cost, values_0)
trans_mat = get_transformation_matrix(values_0)
camera_positions_orig = results['pred_cam_root']
camera_positions_bigger = np.concatenate((camera_positions_orig, np.ones((camera_positions_orig.shape[0], 1))), axis=1)
camera_positions_0_referential = (trans_mat@camera_positions_bigger.T)[:3].T
x_positions, y_positions, z_positions = get_positions_of_base_frame(camera_positions=camera_positions_0_referential - camera_positions_0_referential[0])
# rotation_mat = np.eye(3)
class WaltzRightFootPose:

    def __init__(self, init_time: float, configuration: pink.Configuration):
        """Initialize pose.

        Args:
            init: Initial transform from the wrist frame to the world frame.
        """
        self.init_time = init_time
        base_config = configuration.get_transform_frame_to_world("r_foot")
        base_pos_bigger = np.concatenate(((positions[int(init_time), feet_markers["r_foot"]]).reshape(-1, 3), np.ones((1, 1))), axis=1)
        base_pos_0_referential = (trans_mat@base_pos_bigger.T)[:3].T
        base_pos = (base_pos_0_referential + np.array([x_positions[int(init_time)], y_positions[int(init_time)], z_positions[int(init_time)]])).reshape(-1)
        self.init = base_config.copy()
        self.init.translation = base_pos

    def at(self, t):
        """Get right feet pose at a given time.

        Args:
            t: Time in seconds.
        """
        T = self.init.copy()
        position_bigger = np.concatenate(((positions[int(t), feet_markers["r_foot"]]).reshape(-1, 3), np.ones((1, 1))), axis=1)
        position_0_referential = (trans_mat@position_bigger.T)[:3].T
        position = (position_0_referential + np.array([x_positions[int(t)], y_positions[int(t)], z_positions[int(t)]])).reshape(-1)
        # R = T.rotation
        # R = np.dot(R, pin.utils.rpyToMatrix(0.0, 0.0, np.pi / 2))
        # R = np.dot(R, pin.utils.rpyToMatrix(0.0, -np.pi, 0.0))
        # T.rotation = R
        T.translation = position
        return T
    
class WaltzLeftFootPose:

    def __init__(self, init_time: float, configuration: pink.Configuration):
        """Initialize pose.

        Args:
            init: Initial transform from the wrist frame to the world frame.
        """
        self.init_time = init_time
        base_config = configuration.get_transform_frame_to_world("l_foot")
        base_pos_bigger = np.concatenate(((positions[int(init_time), feet_markers["l_foot"]]).reshape(-1, 3), np.ones((1, 1))), axis=1)
        base_pos_0_referential = (trans_mat@base_pos_bigger.T)[:3].T
        base_pos = (base_pos_0_referential + np.array([x_positions[int(init_time)], y_positions[int(init_time)], z_positions[int(init_time)]])).reshape(-1)
        self.init = base_config.copy()
        self.init.translation = base_pos

    def at(self, t):
        """Get right feet pose at a given time.

        Args:
            t: Time in seconds.
        """
        T = self.init.copy()
        position_bigger = np.concatenate(((positions[int(t), feet_markers["l_foot"]]).reshape(-1, 3), np.ones((1, 1))), axis=1)
        position_0_referential = (trans_mat@position_bigger.T)[:3].T
        position = (position_0_referential + np.array([x_positions[int(t)], y_positions[int(t)], z_positions[int(t)]])).reshape(-1)
        # R = T.rotation
        # R = np.dot(R, pin.utils.rpyToMatrix(0.0, 0.0, np.pi / 2))
        # R = np.dot(R, pin.utils.rpyToMatrix(0.0, -np.pi, 0.0))
        # T.rotation = R
        T.translation = position
        return T
    
class WaltzPelvisPose:

    def __init__(self, init_time: float, configuration: pink.Configuration):
        """Initialize pose.

        Args:
            init: Initial transform from the wrist frame to the world frame.
        """
        self.init_time = init_time
        base_config = configuration.get_transform_frame_to_world("pelvis")
        base_pos_bigger = np.concatenate(((positions[int(init_time), pelvis_markers["pelvis"]]).reshape(-1, 3), np.ones((1, 1))), axis=1)
        base_pos_0_referential = (trans_mat@base_pos_bigger.T)[:3].T
        base_pos = (base_pos_0_referential + np.array([x_positions[int(init_time)], y_positions[int(init_time)], z_positions[int(init_time)]])).reshape(-1)
        self.init = base_config.copy()
        self.init.translation = base_pos

    def at(self, t):
        """Get pelvis pose at a given time.

        Args:
            t: Time in seconds.
        """
        T = self.init.copy()
        position_bigger = np.concatenate(((positions[int(t), pelvis_markers["pelvis"]]).reshape(-1, 3), np.ones((1, 1))), axis=1)
        position_0_referential = (trans_mat@position_bigger.T)[:3].T
        position = (position_0_referential + np.array([x_positions[int(t)], y_positions[int(t)], z_positions[int(t)]])).reshape(-1)
        # R = T.rotation
        # R = np.dot(R, pin.utils.rpyToMatrix(0.0, 0.0, np.pi / 2))
        # R = np.dot(R, pin.utils.rpyToMatrix(0.0, -np.pi, 0.0))
        # T.rotation = R
        T.translation = position
        return T
    
class WaltzHeadPose:

    def __init__(self, init_time: float, configuration: pink.Configuration):
        """Initialize pose.

        Args:
            init: Initial transform from the wrist frame to the world frame.
        """
        self.init_time = init_time
        base_config = configuration.get_transform_frame_to_world("head")
        base_pos_bigger = np.concatenate(((positions[int(init_time), head_markers["head"]]).reshape(-1, 3), np.ones((1, 1))), axis=1)
        base_pos_0_referential = (trans_mat@base_pos_bigger.T)[:3].T
        base_pos = (base_pos_0_referential + np.array([x_positions[int(init_time)], y_positions[int(init_time)], z_positions[int(init_time)]])).reshape(-1)
        self.init = base_config.copy()
        self.init.translation = base_pos

    def at(self, t):
        """Get head pose at a given time.

        Args:
            t: Time in seconds.
        """
        T = self.init.copy()
        position_bigger = np.concatenate(((positions[int(t), head_markers["head"]]).reshape(-1, 3), np.ones((1, 1))), axis=1)
        position_0_referential = (trans_mat@position_bigger.T)[:3].T
        position = (position_0_referential + np.array([x_positions[int(t)], y_positions[int(t)], z_positions[int(t)]])).reshape(-1)
        # R = T.rotation
        # R = np.dot(R, pin.utils.rpyToMatrix(0.0, 0.0, np.pi / 2))
        # R = np.dot(R, pin.utils.rpyToMatrix(0.0, -np.pi, 0.0))
        # T.rotation = R
        T.translation = position
        return T

if __name__ == "__main__":
    robot = load_robot_description(
        "atlas_v4_description", root_joint=pin.JointModelFreeFlyer()
    )

    # Initialize visualization
    viz = start_meshcat_visualizer(robot)
    right_foot_frame = viz.viewer["right_foot_pose"]
    left_foot_frame = viz.viewer["left_foot_pose"]
    pelvis_frame = viz.viewer["pelvis_pose"]
    head_frame = viz.viewer["head_pose"]
    meshcat_shapes.frame(right_foot_frame)
    meshcat_shapes.frame(left_foot_frame)
    meshcat_shapes.frame(pelvis_frame)
    meshcat_shapes.frame(head_frame)

    # Set initial robot configuration
    configuration = pink.Configuration(robot.model, robot.data, robot.q0)
    viz.display(configuration.q)

    # Tasks initialization for IK
    left_foot_task = FrameTask(
        "l_foot",
        position_cost=1.0,
        orientation_cost=1.0,
    )
    right_foot_task = FrameTask(
        "r_foot",
        position_cost=1.0,
        orientation_cost=1.0,
    )
    pelvis_task = FrameTask(
        "pelvis",
        position_cost=1.0e-1,
        orientation_cost=0.0,
    )
    head_task = FrameTask(
        "head",
        position_cost=1.0e-1,
        orientation_cost=0.0,
    )
    posture_task = PostureTask(
        cost=1.0e-1,  # [cost] / [rad]
    )

    tasks = [
        left_foot_task,
        pelvis_task,
        right_foot_task,
        # posture_task,
    ]

    # Task target specifications
    left_foot_task.set_target(
        configuration.get_transform_frame_to_world("l_foot")
    )
    right_foot_task.set_target(
        configuration.get_transform_frame_to_world("r_foot")
    )
    pelvis_task.set_target(
        configuration.get_transform_frame_to_world("pelvis")
    )
    head_task.set_target(
        configuration.get_transform_frame_to_world("head")
    )

    posture_task.set_target_from_configuration(configuration)

    right_foot_pose = WaltzRightFootPose(init_frame, configuration)
    left_foot_pose = WaltzLeftFootPose(init_frame, configuration)
    pelvis_pose = WaltzPelvisPose(init_frame, configuration)
    head_pose = WaltzHeadPose(init_frame, configuration)

    # Select QP solver
    solver = qpsolvers.available_solvers[0]
    if "quadprog" in qpsolvers.available_solvers:
        solver = "quadprog"

    # rate = RateLimiter(frequency=200.0, warn=False)
    # dt = rate.period
    dt = 1.0
    t = init_frame  # [s]
    while True:
        # Update task targets

        right_foot_task.set_target(right_foot_pose.at(t))
        left_foot_task.set_target(left_foot_pose.at(t))
        pelvis_task.set_target(pelvis_pose.at(t))
        head_task.set_target(head_pose.at(t))
        right_foot_frame.set_transform(right_foot_pose.at(t).np)
        left_foot_frame.set_transform(left_foot_pose.at(t).np)
        pelvis_frame.set_transform(pelvis_pose.at(t).np)
        head_frame.set_transform(head_pose.at(t).np)

        # Compute velocity and integrate it into next configuration
        velocity = solve_ik(configuration, tasks, dt, solver=solver)
        configuration.integrate_inplace(velocity, dt)

        # Visualize result at fixed FPS
        viz.display(configuration.q)
        t += dt
        if t > 411.0:
            t = init_frame
        time.sleep(0.03)