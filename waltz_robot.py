#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: MIT
# Authors: Charles Monte and Constantin Vaillant-Tenzer

# Description: This script creates a video of a robot dancing to a given music file.
# The robot's movements are generated using a pre-trained model and the music's BPM.
# The video is created by combining the robot's movements with a background image and the music.
# The video is saved to a file with the specified name.
# Usage: python waltz_robot.py --music_file_path <path_to_music_file> --record_video_path <path_to_save_video> --background_image_path <path_to_background_image> --transformation_values <transformation_values> --movement_file <path_to_movement_file> --credits_text <credits_text> --init_frame <init_frame> --frames_cut_end <frames_cut_end> --nb_turns_in_vid <nb_turns_in_vid> --robot_name <robot_name>
# All arguments are optional and have default values that are the one we used to create the demonstration video.

# Import necessary libraries
import argparse
import os
import subprocess
import numpy as np
import cv2
from src.robot_waltz_from_niki import animate_robot_dancing
from src.bpm_detection import get_bpm
from src.music_lenght_detection import music_lenght


def create_video_robot(movement_file, audio_file, background_image, output_file, credits_text, robot_name="atlas_v4_description", init_frame=30, frames_cut_end=55, nb_turns_in_vid=4, transformation_values = [np.pi, 0., -np.pi/2, 0., 0., 1., 2., 2., 2.]):

    def create_video(audio_file, background_image, output_file, credits_text):
        frame_folder = 'frames/'
        mus_length = music_lenght(audio_file)

        # Load background image
        background = cv2.imread(background_image)
        height, width, _ = background.shape

        # Calculate the frame rate to match the video length to the music length
        frame_rate = 30

        # Create a video writer with the calculated frame rate
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter('temp_video.mp4', fourcc, frame_rate, (width, height))
        
        # Write animation frames to video
        for i in range(len(os.listdir(frame_folder))) :
            if i % (len(os.listdir(frame_folder)) // (frame_rate * mus_length)) == 0:
                frame = cv2.imread(f'frames/frame_{i:04d}.png')
                frame_resized = cv2.resize(frame, (width, height))
                combined_frame = cv2.addWeighted(background, 0.5, frame_resized, 0.5, 0)
                video_writer.write(combined_frame)

        video_writer.release()

        #  Add audio and credits using ffmpeg

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
    
    bpm = get_bpm(audio_file)
    mus_length = music_lenght(audio_file)

    # Animate the robot dancing
    animate_robot_dancing(movement_file, robot_name, bpm, init_frame, frames_cut_end, nb_turns_in_vid, transformation_values, mus_length)

    # Create a video with the robot dancing to the music
    create_video(audio_file, background_image, output_file, credits_text)


if __name__ == '__main__':

    # Create frames directory if it does not exist
    if not os.path.exists('frames'):
        os.makedirs('frames')

    parser = argparse.ArgumentParser(
        description='Create a video of a robot dancing to a given music file.')
    
    parser.add_argument('music_file_path', 
                        type=str, 
                        default='Chostakovitch_Kitaenko_w2.mp3',
                        help='Path to the music file')
    
    parser.add_argument('record_video_path',
                        type=str,
                        default='robot_waltz.mp4',
                        help='Path to save the video file')
    
    parser.add_argument('background_image_path',
                        type=str,
                        default='ballroom.jpg',
                        help='Path to the background image')
    
    parser.add_argument('transformation_values',
                        type=list,
                        default=[np.pi, 0., -np.pi/2, 0., 0., 1., 2., 2., 2.],
                        help='Transformation values for the robot')
    
    parser.add_argument('movement_file',
                        type=str,
                        default='valse_constantin.pt',
                        help='Path to the robot movement file')
    
    parser.add_argument('credits_text',
                        type=str,
                        default='A video realized by Constantin Vaillant-Tenzer and Charles Monte' + '\n' \
                                + 'Music: Chostakovitch, waltz #2 - Directed by D. Kitaenko' + '\n',
                        help='Text to display in the credits')

    parser.add_argument('init_frame',
                        type=int,
                        default=30,
                        help='Initial frame')
    
    parser.add_argument('frames_cut_end',
                        type=int,
                        default=55,
                        help='Frames to cut')
    parser.add_argument('nb_turns_in_vid',
                        type=int,
                        default=4,
                        help='Number of turns in the video')
    parser.add_argument('robot_name',
                        type=str,
                        default='atlas_v4_description',
                        help='Robot name')
    
    args = parser.parse_args()

    create_video_robot(args.movement_file, args.music_file_path, args.background_image_path, args.record_video_path, args.credits_text, args.robot_name, args.init_frame, args.frames_cut_end, args.nb_turn, args.transformation_values)
