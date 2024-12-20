# Waltz Robot Animation

This folder contains a collection of scripts for generating waltz robotic movements using inverse kinematics. The goal is to create a video showcasing these animations.

## Files

- `robot_waltz_animation_drc.py`: This script tests the waltz animation on the star robot from Boston Dynamics, Atlas.
- `robot_waltz_animation_other_robots.py`: This script explores waltz animations on other robots apart from Atlas.
- `robot_waltz_animation.py` : This is the original file where the robot's base was stuck to the origin which we have shown during the poster session.
- `robot_waltz_perso.py` : Generate (without creating a video) the best simulation we could do using Pose2Sim.
- `optimal_rotation_matrix.py` : Generate and show the good rotation matrix in order to get the right kinematics from the output of Pose2Sim (files position_joints.mot and rotation_joints.mot). This code has also been incorporated in the main pipeline.

## Conclusion
Since the animation was very noisy, we decided to use the Kinematics obtained through the niki model.