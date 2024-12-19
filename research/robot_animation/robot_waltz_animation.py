import numpy as np
import pink
import pinocchio as pin
from pink.visualization import start_meshcat_visualizer
from pink.tasks import FrameTask, PostureTask, ComTask, JointCouplingTask
from pink import solve_ik
from robot_descriptions.loaders.pinocchio import load_robot_description
import pandas as pd
import time
from pydub import AudioSegment
import cv2
import subprocess
import os
import scipy.signal


class RobotWaltzAnimation:
    CONFIGURATION_DT = 0.0344828

    def __init__(self, robot_description, position_file):
        # Load the robot description
        self.robot = load_robot_description(robot_description)
        self.configuration = pink.Configuration(self.robot.model, self.robot.data, self.robot.q0)

        # Get the names of the robot frames
        #robot_frame_names = [frame for frame in self.robot.model.frames]    
        #robot_frame_names = [frame for frame in robot_frame_names if frame != "universe"]
        robot_frame_names = ['root_joint', 'pelvis', 'back_bkz', 'ltorso', 'back_bky', 'mtorso', 'back_bkx', 'utorso', 'l_arm_shz', 'l_clav', 'l_arm_shx', 'l_scap', 'l_arm_ely', 'l_uarm', 'l_arm_elx', 'l_larm', 'l_arm_wry', 'l_ufarm', 'l_arm_wrx', 'l_lfarm', 'l_arm_wry2', 'l_hand', 'neck_ry', 'head', 'r_arm_shz', 'r_clav', 'r_arm_shx', 'r_scap', 'r_arm_ely', 'r_uarm', 'r_arm_elx', 'r_larm', 'r_arm_wry', 'r_ufarm', 'r_arm_wrx', 'r_lfarm', 'r_arm_wry2', 'r_hand', 'l_leg_hpz', 'l_uglut', 'l_leg_hpx', 'l_lglut', 'l_leg_hpy', 'l_uleg', 'l_leg_kny', 'l_lleg', 'l_leg_aky', 'l_talus', 'l_leg_akx', 'l_foot', 'r_leg_hpz', 'r_uglut', 'r_leg_hpx', 'r_lglut', 'r_leg_hpy', 'r_uleg', 'r_leg_kny', 'r_lleg', 'r_leg_aky', 'r_talus', 'r_leg_akx', 'r_foot']

        # Define tasks for Pink
        self.tasks = {frame: FrameTask(frame, position_cost=1e-3, orientation_cost=1e-3) for frame in robot_frame_names}
        self.tasks["posture"] = PostureTask(cost=1e-3)
        self.tasks["l_foot"].position_cost = 10.0
        self.tasks["r_foot"].position_cost = 10.0
        self.tasks["l_foot"].orientation_cost = 10.0
        self.tasks["r_foot"].orientation_cost = 10.0
        self.tasks["head"].position_cost = 10.0
        self.tasks["head"].orientation_cost = 10.0
        self.tasks["pelvis"].position_cost=5.0
        self.tasks["pelvis"].orientation_cost=5.0
        # Set the initial target for the root_joint
        root_joint_pose = self.configuration.get_transform_frame_to_world("root_joint").copy()
        self.tasks["root_joint"] = FrameTask("root_joint", position_cost=1e-3, orientation_cost=1e-3)
        self.tasks["root_joint"].set_target(root_joint_pose)

        


        '''# Joint coupling task
        head_pelvis_coupling = JointCouplingTask(
        ["head", "back_bkz"],
        [1.0, -1.0],
        100.0,
        self.configuration,
        lm_damping=1e-7,
        )

        
       
       # Set initial targets for tasks
        pelvis_pose = self.configuration.get_transform_frame_to_world("pelvis").copy()
        self.tasks["pelvis"].set_target(pelvis_pose)

        transform_l_foot_target_to_init = pin.SE3(np.eye(3), np.array([0.1, 0.0, 0.0]))
        transform_r_foot_target_to_init = pin.SE3(np.eye(3), np.array([-0.1, 0.0, 0.0]))

        self.tasks["l_foot"].set_target(
            self.configuration.get_transform_frame_to_world("l_foot") * transform_l_foot_target_to_init
        )
        self.tasks["r_foot"].set_target(
            self.configuration.get_transform_frame_to_world("r_foot") * transform_r_foot_target_to_init
        )

        self.tasks["head"].set_target(
            self.configuration.get_transform_frame_to_world("head")
        )

        self.tasks["posture"].set_target_from_configuration(self.configuration)
        #self.tasks["head_pelvis_coupling"] = head_pelvis_coupling 
        '''


        # Add CoM task
        #self.tasks["com"] = ComTask(self.robot)

        # Set posture task to the robot's default configuration
        self.tasks["posture"].set_target(self.configuration.q)

        # Read position data
        self.data_pos = pd.read_csv(position_file, delimiter='\t', skiprows=3)
        self.time_col = self.data_pos['Time']

        # List of robot joints (these are the joint names)
        self.joint_names = self.robot.model.names
        self.robot_joints = [joint for joint in self.joint_names]
        print(f"Robot joints: {self.robot_joints}")

        # List of marker names in the .trc file (example: X1, Y1, Z1, ..., X22, Y22, Z22)
        self.marker_columns = [
            "X1", "Y1", "Z1", "X2", "Y2", "Z2", "X3", "Y3", "Z3", "X4", "Y4", "Z4", "X5", "Y5", "Z5",
            "X6", "Y6", "Z6", "X7", "Y7", "Z7", "X8", "Y8", "Z8", "X9", "Y9", "Z9", "X10", "Y10", "Z10",
            "X11", "Y11", "Z11", "X12", "Y12", "Z12", "X13", "Y13", "Z13", "X14", "Y14", "Z14", "X15",
            "Y15", "Z15", "X16", "Y16", "Z16", "X17", "Y17", "Z17", "X18", "Y18", "Z18", "X19", "Y19",
            "Z19", "X20", "Y20", "Z20", "X21", "Y21", "Z21", "X22", "Y22", "Z22"
        ]

         # Read rotation data
        self.data_rot = pd.read_csv('rotation_joints.mot', delimiter='\t', skiprows=10)

        # Identify joints with their 3D positions
        rotation_joint_names = [
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

        # Mapping from joints to markers
        self.joint_marker_map = {
            "back_bkz": ["X1", "Y1", "Z1"],  # Hip
            "r_leg_hpz": ["X2", "Y2", "Z2"],  # RHip
            "r_leg_kny": ["X3", "Y3", "Z3"],  # RKnee
            "r_leg_aky": ["X4", "Y4", "Z4"],  # RAnkle
            "r_leg_mtp": ["X5", "Y5", "Z5"],  # RBigToe
            "r_leg_mtp2": ["X6", "Y6", "Z6"],  # RSmallToe
            "r_leg_heel": ["X7", "Y7", "Z7"],  # RHeel
            "l_leg_hpz": ["X8", "Y8", "Z8"],  # LHip
            "l_leg_kny": ["X9", "Y9", "Z9"],  # LKnee
            "l_leg_aky": ["X10", "Y10", "Z10"],  # LAnkle
            "l_leg_mtp": ["X11", "Y11", "Z11"],  # LBigToe
            "l_leg_mtp2": ["X12", "Y12", "Z12"],  # LSmallToe
            "l_leg_heel": ["X13", "Y13", "Z13"],  # LHeel
            "neck_ry": ["X14", "Y14", "Z14"],  # Neck
            "head": ["X15", "Y15", "Z15"],  # Head
            "nose": ["X16", "Y16", "Z16"],  # Nose
            "r_arm_shz": ["X17", "Y17", "Z17"],  # RShoulder
            "r_arm_ely": ["X18", "Y18", "Z18"],  # RElbow
            "r_arm_wry": ["X19", "Y19", "Z19"],  # RWrist
            "l_arm_shz": ["X20", "Y20", "Z20"],  # LShoulder
            "l_arm_ely": ["X21", "Y21", "Z21"],  # LElbow
            "l_arm_wry": ["X22", "Y22", "Z22"],  # LWrist
        }

        # Mapping from joints to rotations
        self.joint_rotation_map = {
            "pelvis": ["pelvis_tilt", "pelvis_list", "pelvis_rotation"],
            "r_hip": ["hip_flexion_r", "hip_adduction_r", "hip_rotation_r"],
            "r_knee": ["knee_angle_r", "knee_angle_r_beta"],
            "r_ankle": ["ankle_angle_r", "subtalar_angle_r"],
            "l_hip": ["hip_flexion_l", "hip_adduction_l", "hip_rotation_l"],
            "l_knee": ["knee_angle_l", "knee_angle_l_beta"],
            "l_ankle": ["ankle_angle_l", "subtalar_angle_l"],
            "spine": ["L5_S1_Flex_Ext", "L5_S1_Lat_Bending", "L5_S1_axial_rotation"],
            "neck": ["neck_flexion", "neck_bending", "neck_rotation"],
            "r_shoulder": ["arm_flex_r", "arm_add_r", "arm_rot_r"],
            "r_elbow": ["elbow_flex_r", "pro_sup_r"],
            "r_wrist": ["wrist_flex_r", "wrist_dev_r"],
            "l_shoulder": ["arm_flex_l", "arm_add_l", "arm_rot_l"],
            "l_elbow": ["elbow_flex_l", "pro_sup_l"],
            "l_wrist": ["wrist_flex_l", "wrist_dev_l"],
        }

        # Mapping from feet to markers
        self.feet_markers = {"l_foot": ["X11", "Y11", "Z11"], "r_foot": ["X5", "Y5", "Z5"]}
        self.head_markers = {"head": ["X15", "Y15", "Z15"]}
        self.pelvis_markers = {"pelvis": ["X1", "Y1", "Z1"]}
        self.head_rotation = {"head": ["neck_flexion", "neck_bending", "neck_rotation"]}
        self.feet_rotations = {"l_foot": ["ankle_angle_l", "subtalar_angle_l", "mtp_angle_l"],
                               "r_foot": ["ankle_angle_r", "subtalar_angle_r", "mtp_angle_r"]}
        self.pelvis_roations = {"pelvis": ["pelvis_tilt", "pelvis_list", "pelvis_rotation"]}
        
        # Mapping other joints to markers
        #self.other_joints = set(self.joint_marker_map.keys()) - set(self.feet_markers.keys())
        #self.other_joints = list(self.other_joints)
        

        self.validate_joint_mapping()

        # Set the initial configuration for the tasks
        self.configuration = pink.Configuration(self.robot.model, self.robot.data, self.robot.q0)
        for body, task in self.tasks.items():
            if isinstance(task, FrameTask):
                task.set_target(self.configuration.get_transform_frame_to_world(body))

        # Visualize the initial configuration
        self.viz = start_meshcat_visualizer(self.robot)
        self.viz.display(self.configuration.q)

    def validate_joint_mapping(self):
        for joint, markers in self.joint_marker_map.items():
            if len(markers) != 3:
                print(f"Warning: {joint} does not have exactly 3 markers (X, Y, Z).")
            for marker in markers:
                if marker not in self.data_pos.columns:
                    print(f"Error: Marker {marker} for joint {joint} not found in the dataset.")
                else:
                    print(f"Mapping for {joint}: {marker} found.")
        for joint, rotations in self.joint_rotation_map.items():
            if len(rotations) != 3:
                print(f"Warning: {joint} does not have exactly 3 rotations.")
            for rotation in rotations:
                if rotation not in self.data_rot.columns:
                    print(f"Error: Rotation {rotation} for joint {joint} not found in the dataset.")
                else:
                    print(f"Mapping for {joint}: {rotation} found.")

    def extract_initial_positions(self, time_sec):
        time_idx = np.argmin(np.abs(pd.to_numeric(self.data_pos['Time'], errors='coerce') - time_sec))
        initial_positions = {}
        for joint, markers in self.joint_marker_map.items():
            positions = self.data_pos.loc[time_idx, markers].values
            initial_positions[joint] = positions
            print(f"Initial position for {joint}: {positions}")

        unmapped_joints = set(self.robot_joints) - set(self.joint_marker_map.keys())
        default_position = np.zeros(3)
        for joint in unmapped_joints:
            initial_positions[joint] = default_position
            print(f"Default position for unmapped joint {joint}: {default_position}")

        return initial_positions
    
    def extract_initial_rotations(self, time_sec):
        time_idx = np.argmin(np.abs(pd.to_numeric(self.data_pos['Time'], errors='coerce') - time_sec))
        initial_rotations = {}
        for joint, rotations in self.joint_rotation_map.items():
            positions = self.data_rot.loc[time_idx, rotations].values
            initial_rotations[joint] = positions
            print(f"Initial rotation for {joint}: {positions}")

        unmapped_joints = set(self.robot_joints) - set(self.joint_rotation_map.keys())
        default_rotation = np.zeros(3)
        for joint in unmapped_joints:
            initial_rotations[joint] = default_rotation
            print(f"Default rotation for unmapped joint {joint}: {default_rotation}")

        return initial_rotations

    def animate(self, bpm=120.):
        FACTOR_BPM = 13
        dt = FACTOR_BPM / bpm
        animation_frames = []

        n_idx = int((180 + 47) / dt)
        print(f"Number of frames: {n_idx}")

        for idx_c in range(n_idx):
            idx = idx_c % len(self.time_col)
            # Update foot positions
            for foot, markers in self.feet_markers.items():
                position = self.data_pos.loc[idx, markers]
                position_quaternion = pin.SE3.Identity().rotation
                self.tasks[foot].set_target(pin.SE3(position_quaternion, np.array(position)))

            # Update head position
            for head, markers in self.head_markers.items():
                position = self.data_pos.loc[idx, markers]
                position_quaternion = pin.SE3.Identity().rotation
                self.tasks[head].set_target(pin.SE3(position_quaternion, np.array(position)))

            # Update pelvis position
            for pelvis, markers in self.pelvis_markers.items():
                position = self.data_pos.loc[idx, markers]
                position_quaternion = pin.SE3.Identity().rotation
                self.tasks[pelvis].set_target(pin.SE3(position_quaternion, np.array(position)))

            # Update pelvis position based on feet
            l_foot_pos = self.data_pos.loc[idx, self.feet_markers["l_foot"]].values
            r_foot_pos = self.data_pos.loc[idx, self.feet_markers["r_foot"]].values
            pelvis_pos = (l_foot_pos + r_foot_pos) / 2  # Average position of feet
            pelvis_rotation = pin.SE3.Identity().rotation  # Placeholder for rotation
            self.tasks["pelvis"].set_target(pin.SE3(pelvis_rotation, pelvis_pos))

            # Update root_joint position based on pelvis
            root_joint_pos = pelvis_pos + np.array([0.0, 0.0, 0.1])  # Adjust the position as needed
            self.tasks["root_joint"].set_target(pin.SE3(pelvis_rotation, root_joint_pos))

            # Solve inverse kinematics and update configuration
            velocity = solve_ik(self.configuration, self.tasks.values(), dt, solver="quadprog")
            self.configuration.integrate_inplace(velocity, dt)
            print("Tasks included in IK:", [task for task in self.tasks.keys()])

            print(f"Frame {idx_c}, Configuration: {self.configuration.q}")
            self.viz.display(self.configuration.q)
            frame = self.viz.captureImage()
            animation_frames.append(frame[:,:,:3])
            time.sleep(dt)

        return animation_frames



def get_bpm(file_path):
    # Load the audio file
    audio = AudioSegment.from_file(file_path)

    # Convert to mono and get raw data
    audio = audio.set_channels(1)
    samples = np.array(audio.get_array_of_samples())

    # Calculate the envelope of the signal
    envelope = np.abs(scipy.signal.hilbert(samples))

    # Downsample the envelope to reduce computation
    downsample_factor = 100
    envelope = scipy.signal.decimate(envelope, downsample_factor)

    # Calculate autocorrelation
    autocorr = np.correlate(envelope, envelope, mode='full')
    autocorr = autocorr[len(autocorr) // 2:]

    # Find peaks in the autocorrelation
    peaks, _ = scipy.signal.find_peaks(autocorr, distance=audio.frame_rate // 2)

    # Calculate the intervals between peaks
    intervals = np.diff(peaks)

    # Calculate BPM
    bpm = 60.0 / (np.mean(intervals) * downsample_factor / audio.frame_rate)

    return bpm


def create_video(animation_frames, audio_file, background_image, output_file, credits_text):
    # Load background image
    background = cv2.imread(background_image)
    height, width, _ = background.shape

    # Create a video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter('temp_video.mp4', fourcc, 30, (width, height))

    # Write animation frames to video
    for frame in animation_frames:
        frame_resized = cv2.resize(frame, (width, height))
        combined_frame = cv2.addWeighted(background, 0.5, frame_resized, 0.5, 0)
        video_writer.write(combined_frame)

    video_writer.release()

    # Add audio and credits using ffmpeg

    # Create a temporary credits image
    credits_image = 'credits.png'
    credits = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.putText(credits, credits_text, (10, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imwrite(credits_image, credits)

    # Combine video, audio, and credits using ffmpeg
    command = [
        'ffmpeg',
        '-i', 'temp_video.mp4',
        '-i', audio_file,
        '-loop', '1', '-t', '4', '-i', credits_image,
        '-filter_complex', '[0:v][2:v]concat=n=2:v=1:a=0',
        '-c:v', 'libx264',
        '-c:a', 'aac',
        '-shortest',
        output_file
    ]
    subprocess.run(command)

    # Clean up temporary files
    os.remove('temp_video.mp4')
    os.remove(credits_image)


# Example usage
file_path = 'Chostakovitch_Kitaenko_w2.mp3'
bpm = get_bpm(file_path)
print(f"BPM: {bpm}")

# Example usage
robot_animation = RobotWaltzAnimation("atlas_v4_description", 'position_joints.trc')
animation_frames = robot_animation.animate(bpm=187)

# Example usage
create_video(animation_frames, 'Chostakovitch_Kitaenko_w2.mp3', 'ballroom.jpg', 'robot_waltz.mp4', 'A video realized by Constantin Vaillant-Tenzer and Charles Monte \n'
              + 'Music: Chostakovitch, waltz #2 - D. Kitaenko')