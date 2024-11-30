import pinocchio as pin
import robot_descriptions
from robot_descriptions.loaders.pinocchio import load_robot_description
import numpy as np

import show_in_meshcat

# Show in meshcat the robot model

robots_to_test = [
    "atlas_drc_description", "atlas_v4_description", "draco3_description",
    "ergocub_description", "g1_description", "g1_mj_description", "h1_description", "h1_mj_description",
    "icub_description", "jaxon_description", "jvrc_description", "jvrc_mj_description", "op3_mj_description",
    "r2_description", "romeo_description", "sigmaban_description", "talos_description", "talos_mj_description",
    "valkyrie_description"
]


# Benchmark the robot's agility of the feet
def benchmark_feet_agility(robot):
    # Assuming the robot model has joints for the feet
    feet_joints = ["left_foot", "right_foot"]
    agility_scores = {}


    '''
    for joint in feet_joints:
        joint_id = robot.model.getJointId(joint)
        print( "Joint ID: ", joint_id)

        if joint_id != pin.JointModelFreeFlyer.id and 0 <= joint_id < len(robot.model.upperPositionLimit):
            joint_limits = robot.model.upperPositionLimit[joint_id] - robot.model.lowerPositionLimit[joint_id]
            agility_scores[joint] = joint_limits
        else:
            agility_scores[joint] = 0.0
        '''

    return agility_scores


for robot_name in robots_to_test:
    # Load the robot model
    try:
        robot = load_robot_description(robot_name)
        print(f"Robot successfully loaded as {robot}")
    except IndexError as e:
        print(f"Failed to load {robot_name}: {e}")
        

    # Get the names of all the joints
    joint_names = robot.model.names
    print(f"Joint names for {robot_name}: {joint_names}")
    
    # Get the number of joints
    num_joints = len(joint_names)
    print(f"Number of joints for {robot_name}: {num_joints}")
    
    # Get the joint limits
    joint_limits = {}
    for joint in joint_names:
        print("Joint: ", joint)
        joint_id = robot.model.getJointId(joint)
        if joint_id != pin.JointModelFreeFlyer.id and 0 <= joint_id < len(robot.model.upperPositionLimit):
            lower_limit = robot.model.lowerPositionLimit[joint_id]
            upper_limit = robot.model.upperPositionLimit[joint_id]
            joint_limits[joint] = (lower_limit, upper_limit)
        else:
            joint_limits[joint] = (None, None)
    
    #print(f"Joint limits for {robot_name}: {joint_limits}")
    agility_scores = benchmark_feet_agility(robot)
    #print("Feet agility scores of ", robot_name, ": ", agility_scores)



