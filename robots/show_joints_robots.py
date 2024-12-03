#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: MIT license
# Copyright 2024 Constantin Vaillant Tenzer


"""
Load a robot description - featuring only the joints, specified from the command line, in RoboMeshCat.

This example uses RoboMeshCat: https://github.com/petrikvladimir/RoboMeshCat
"""

import argparse
import robomeshcat
import numpy as np

from robot_descriptions.loaders.robomeshcat import load_robot_description

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name", help="name of the robot description")
    args = parser.parse_args()

    try:
        robot = load_robot_description(args.name)
    except ModuleNotFoundError:
        robot = load_robot_description(f"{args.name}_description")

    # Initialize the MeshCat scene
    scene = robomeshcat.Scene()

    # Access joint placements
    joints = robot._model.joints  # Access joints
    print(dir(joints))
    joint_placements = robot._model.jointPlacements  # Joint transformations
    print(dir(joint_placements))

    for i, joint in enumerate(joints):
        if i == 0:  # Skip the root joint (world or fixed base)
            continue
        
        joint_name = joint.name
        joint_position = joint_placements[i].translation  # Get joint position
        
        # Visualize the joint as a sphere
        scene.add_sphere(f"joint_{joint_name}", 0.02, position=joint_position, color=[1, 0, 0])

    scene.render()
    scene.show()
