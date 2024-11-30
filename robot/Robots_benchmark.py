import pinocchio as pin
import robot_descriptions
from robot_descriptions.loaders.pinocchio import load_robot_description

import show_in_meshcat

# Show in meshcat the robot model

robots_to_test = [
    "atlas_drc_description", "atlas_v4_description", "berkeley_humanoid_description", "draco3_description",
    "ergocub_description", "g1_description", "g1_mj_description", "h1_description", "h1_mj_description",
    "icub_description", "jaxon_description", "jvrc_description", "jvrc_mj_description", "op3_mj_description",
    "r2_description", "romeo_description", "sigmaban_description", "talos_description", "talos_mj_description",
    "valkyrie_description"
]

for robot in robots_to_test:
    show_in_meshcat.robot = load_robot_description(robot)
    #show_in_meshcat.show_robot_model()


# Load the robot model
robot = load_robot_description("atlas_drc_description")
print(f"Robot successfully loaded as {robot}")


# Benchmark the robot's agility of the feet
def benchmark_feet_agility(robot):
    # Assuming the robot model has joints for the feet
    feet_joints = ["left_foot", "right_foot"]
    agility_scores = {}

    for joint in feet_joints:
        joint_id = robot.model.getJointId(joint)
        joint_limits = robot.model.upperPositionLimit[joint_id -1] - robot.model.lowerPositionLimit[joint_id - 1]
        agility_scores[joint] = joint_limits

    return agility_scores


#agility_scores = benchmark_feet_agility(robot)
#print(f"Feet agility scores: {agility_scores}")


'''# Load the robot model
import argparse

from robot_descriptions.loaders.pinocchio import load_robot_description

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name", help="name of the robot description")
    args = parser.parse_args()

    try:
        robot = load_robot_description(args.name)
    except ModuleNotFoundError:
        robot = load_robot_description(f"{args.name}_description")

    print(f"Robot successfully loaded as {robot}")'''
