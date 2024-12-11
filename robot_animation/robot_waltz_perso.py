#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 Inria

"""DRACO 3 humanoid standing on two feet and reaching with a hand."""

import numpy as np
import pandas as pd
import pinocchio as pin
import qpsolvers
import time
from scipy.optimize import fmin_bfgs

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

feet_markers = {"l_foot": ["X11", "Y11", "Z11"], "r_foot": ["X5", "Y5", "Z5"]}
pelvis_markers = {"pelvis": ["X1", "Y1", "Z1"]}
data_pos = pd.read_csv('position_joints.trc', delimiter='\t', skiprows=3)
init_frame = 275.0
head_pose = np.array(data_pos.loc[275:, ["X15", "Y15", "Z15"]])
pelvis_pose = np.array(data_pos.loc[275:, ["X1", "Y1", "Z1"]])
r_foot_pose = np.array(data_pos.loc[275:, ["X5", "Y5", "Z5"]])
l_foot_pose = np.array(data_pos.loc[275:, ["X11", "Y11", "Z11"]])

def get_rotation_matrix(rot_angles):
    alpha = rot_angles[0]; beta = rot_angles[1]; gamma = rot_angles[2]
    ca = np.cos(alpha); sa = np.sin(alpha)
    cb = np.cos(beta); sb = np.sin(beta)
    cg = np.cos(gamma); sg = np.sin(gamma)
    rot_mat = np.array([[ca*cb, ca*sb*sg - sa*cg, ca*sb*cg + sa*sg],
                        [sa*cb, sa*sb*sg + ca*cg, sa*sb*cg - ca*sg],
                        [-sb, cb*sg, cb*cg]])
    return rot_mat

def cost(rot_angles):
    # The cost is the absolute mean z-coordinate of the head
    rot_mat = get_rotation_matrix(rot_angles)
    rotated_head = head_pose@rot_mat
    rotated_pelvis = pelvis_pose@rot_mat
    rotated_r_foot = r_foot_pose@rot_mat
    rotated_l_foot = l_foot_pose@rot_mat
    cost = np.sum(rotated_head[:, 2] - 1.5)**2
    # cost += np.sum(rotated_l_foot[:, 2])**2
    cost += np.sum(rotated_r_foot[:, 2])**2
    # cost = np.var(rotated_pelvis[:, 2])
    # cost += np.mean(rotated_feet[:, 1] - rotated_head[:, 1])**2
    return cost

angles_0 = [0.0, 0.0, 0.0]
angles_opt = fmin_bfgs(cost, angles_0)
rotation_mat = get_rotation_matrix(angles_opt)
# rotation_mat = np.eye(3)
class WaltzRightFootPose:

    def __init__(self, init_time: float, configuration: pink.Configuration):
        """Initialize pose.

        Args:
            init: Initial transform from the wrist frame to the world frame.
        """
        self.init_time = init_time
        base_config = configuration.get_transform_frame_to_world("r_foot")
        base_pos = (np.array(data_pos.loc[init_time, feet_markers["r_foot"]]).reshape(-1, 3)@rotation_mat).reshape(-1)
        self.init = base_config.copy()
        self.init.translation = base_pos

    def at(self, t):
        """Get right feet pose at a given time.

        Args:
            t: Time in seconds.
        """
        T = self.init.copy()
        position = (np.array(data_pos.loc[t, feet_markers["r_foot"]]).reshape(-1, 3)@rotation_mat).reshape(-1)
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
        base_pos = (np.array(data_pos.loc[init_time, feet_markers["l_foot"]]).reshape(-1, 3)@rotation_mat).reshape(-1)
        self.init = base_config.copy()
        self.init.translation = base_pos

    def at(self, t):
        """Get right feet pose at a given time.

        Args:
            t: Time in seconds.
        """
        T = self.init.copy()
        position = (np.array(data_pos.loc[t, feet_markers["l_foot"]]).reshape(-1, 3)@rotation_mat).reshape(-1)
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
        base_pos = (np.array(data_pos.loc[init_time, pelvis_markers["pelvis"]]).reshape(-1, 3)@rotation_mat).reshape(-1)
        self.init = base_config.copy()
        self.init.translation = base_pos

    def at(self, t):
        """Get right feet pose at a given time.

        Args:
            t: Time in seconds.
        """
        T = self.init.copy()
        position = (np.array(data_pos.loc[t, pelvis_markers["pelvis"]]).reshape(-1, 3)@rotation_mat).reshape(-1)
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
    meshcat_shapes.frame(right_foot_frame)
    meshcat_shapes.frame(left_foot_frame)
    meshcat_shapes.frame(pelvis_frame)


    # Set initial robot configuration
    configuration = pink.Configuration(robot.model, robot.data, robot.q0)
    viz.display(configuration.q)

    # Tasks initialization for IK
    left_foot_task = FrameTask(
        "l_foot",
        position_cost=1.0,
        orientation_cost=1.0,
    )
    pelvis_task = FrameTask(
        "pelvis",
        position_cost=1.0,
        orientation_cost=0.0,
    )
    right_foot_task = FrameTask(
        "r_foot",
        position_cost=1.0,
        orientation_cost=1.0,
    )
    posture_task = PostureTask(
        cost=1e-1,  # [cost] / [rad]
    )

    tasks = [
        left_foot_task,
        pelvis_task,
        right_foot_task,
        # posture_task,
    ]

    # Task target specifications
    pelvis_pose = configuration.get_transform_frame_to_world(
        "pelvis"
    ).copy()
    pelvis_pose.translation[0] += 0.05
    pelvis_task.set_target(pelvis_pose)

    transform_l_ankle_target_to_init = pin.SE3(
        np.eye(3), np.array([0.1, 0.0, 0.0])
    )
    transform_r_ankle_target_to_init = pin.SE3(
        np.eye(3), np.array([-0.1, 0.0, 0.0])
    )

    left_foot_task.set_target(
        configuration.get_transform_frame_to_world("l_foot")
        * transform_l_ankle_target_to_init
    )
    right_foot_task.set_target(
        configuration.get_transform_frame_to_world("r_foot")
        * transform_r_ankle_target_to_init
    )

    pelvis_task.set_target(
        configuration.get_transform_frame_to_world("pelvis")
    )

    posture_task.set_target_from_configuration(configuration)

    right_foot_pose = WaltzRightFootPose(init_frame, configuration)
    left_foot_pose = WaltzLeftFootPose(init_frame, configuration)
    pelvis_pose = WaltzPelvisPose(init_frame, configuration)

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
        right_foot_frame.set_transform(right_foot_pose.at(t).np)
        left_foot_frame.set_transform(left_foot_pose.at(t).np)
        pelvis_frame.set_transform(pelvis_pose.at(t).np)

        # Compute velocity and integrate it into next configuration
        velocity = solve_ik(configuration, tasks, dt, solver=solver)
        configuration.integrate_inplace(velocity, dt)

        # Visualize result at fixed FPS
        viz.display(configuration.q)
        t += dt
        if t > 411.0:
            t = init_frame
        time.sleep(0.03)