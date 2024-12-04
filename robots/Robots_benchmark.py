import pinocchio as pin
import robot_descriptions
from robot_descriptions.loaders.pinocchio import load_robot_description
import numpy as np
import matplotlib.pyplot as plt




# Benchmark the robot's agility of the feet
def benchmark_feet_agility(robot):
    # Assuming the robot model has joints for the feet
    feet_joints = ["left_foot", "right_foot"]
    agility_scores = {}


    # Get the joint limits
    for joint in feet_joints:
        joint_id = robot.model.getJointId(joint)
        print( "Joint ID: ", joint_id)

        if joint_id != pin.JointModelFreeFlyer.id and 0 <= joint_id < len(robot.model.upperPositionLimit):
            joint_limits = robot.model.upperPositionLimit[joint_id] - robot.model.lowerPositionLimit[joint_id]
            agility_scores[joint] = joint_limits
        else:
            agility_scores[joint] = 0.0
        

    return agility_scores

def get_joint_names(robot_name):
    robot = load_robot_description(robot_name)
    joint_names = robot.model.names
    for joint in joint_names:
        print(joint)
    return joint_names

def benchmark_feet_agility():
    robots_to_test = [
    "atlas_drc_description", "atlas_v4_description", "draco3_description",
    "ergocub_description", "g1_description", "g1_mj_description", "h1_description", "h1_mj_description",
    "icub_description", "jaxon_description", "jvrc_description", "jvrc_mj_description", "op3_mj_description",
    "r2_description", "romeo_description", "sigmaban_description", "talos_description", "talos_mj_description",
    "valkyrie_description"
    ]


    # Assuming the robot model has joints for the feet
    feet_joints = ["left_foot", "right_foot"]
    agility_scores_dict = {}

    for robot_name in robots_to_test:
        # Load the robot model
        try:
            robot = load_robot_description(robot_name)
            print(f"Robot successfully loaded as {robot}")
        except IndexError as e:
            print(f"Failed to load {robot_name}: {e}")
        

        # Get the names of all the joints
        joint_names = robot.model.names
        
        
        # Get the number of joints
        num_joints = len(joint_names)
        print(f"Number of joints for {robot_name}: {num_joints}")
        
        # Get the joint limits
        joint_limits = {}
        for joint in joint_names:
            #print("Joint: ", joint)
            joint_id = robot.model.getJointId(joint)
            if joint_id != pin.JointModelFreeFlyer.id and 0 <= joint_id < len(robot.model.upperPositionLimit):
                lower_limit = robot.model.lowerPositionLimit[joint_id]
                upper_limit = robot.model.upperPositionLimit[joint_id]
                joint_limits[joint] = (lower_limit, upper_limit)
            else:
                joint_limits[joint] = (None, None)
        
        print(f"Joint limits for {robot_name}: {joint_limits}")
        agility_scores = benchmark_feet_agility(robot)
        print("Feet agility scores of ", robot_name, ": ", agility_scores)

        agility_scores_dict[robot_name] = agility_scores

    print("Agility scores: ", agility_scores_dict)
    print("best robot: ", max(agility_scores_dict, key=lambda k: sum(agility_scores_dict[k].values())))

    return agility_scores_dict


def visualize_robot_joints(robot_name):
        robot = load_robot_description(robot_name)
        robot.forwardKinematics(robot.q0)
        joint_positions = robot.data.oMi
        joint_names = robot.model.names

        fig, ax = plt.subplots()
        for joint_id, joint_name in enumerate(joint_names):
            if joint_id == 0:  # Skip the universe joint
                continue
            position = joint_positions[joint_id].translation
            ax.plot(position[0], position[1], 'o', label=joint_name)
            ax.text(position[0], position[1], joint_name)

        ax.set_xlabel('X position')
        ax.set_ylabel('Y position')
        ax.set_title(f'Joint positions for {robot_name}')
        ax.legend()
        plt.show()

def visualize_robot_joints_3d(robot_name):
        robot = load_robot_description(robot_name)
        robot.forwardKinematics(robot.q0)
        joint_positions = robot.data.oMi
        joint_names = robot.model.names

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for joint_id, joint_name in enumerate(joint_names):
                if joint_id == 0:  # Skip the universe joint
                    continue
                position = joint_positions[joint_id].translation
                ax.scatter(position[0], position[1], position[2], label=joint_name)
                ax.text(position[0], position[1], position[2], joint_name)

        ax.set_xlabel('X position')
        ax.set_ylabel('Y position')
        ax.set_zlabel('Z position')
        ax.set_title(f'Joint positions for {robot_name}')
        ax.legend()
        plt.show()

        def force_neck_highest_z(robot_name):
            robot = load_robot_description(robot_name)
            robot.forwardKinematics(robot.q0)
            joint_positions = robot.data.oMi
            joint_names = robot.model.names

            neck_joint_id = None
            for joint_id, joint_name in enumerate(joint_names):
                if "neck" in joint_name.lower():
                    neck_joint_id = joint_id
                    break

            if neck_joint_id is None:
                print(f"No neck joint found for {robot_name}")
                return

            highest_z = max(joint_positions, key=lambda x: x.translation[2]).translation[2]
            neck_position = joint_positions[neck_joint_id].translation
            neck_position[2] = highest_z

            print(f"Neck joint {joint_names[neck_joint_id]} forced to highest Z position: {neck_position}")

        def force_robot_vertical(robot_name):
            robot = load_robot_description(robot_name)
            robot.forwardKinematics(robot.q0)
            joint_positions = robot.data.oMi
            joint_names = robot.model.names

            for joint_id, joint_name in enumerate(joint_names):
                if joint_id == 0:  # Skip the universe joint
                    continue
                position = joint_positions[joint_id].translation
                position[2] = max(position[2], 0)  # Ensure Z position is non-negative

            print(f"Robot {robot_name} forced to be vertical")

        force_neck_highest_z("atlas_drc_description")
        force_robot_vertical("atlas_drc_description")

visualize_robot_joints_3d("atlas_drc_description")
