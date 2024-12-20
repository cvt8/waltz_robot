import os
import subprocess
import numpy as np
import cv2
from src.robot_waltz_from_niki import animate_robot_dancing
from src.bpm_detection import get_bpm
from src.music_lenght_detection import music_lenght


def create_video_robot(movement_file, audio_file, background_image, output_file, credits_text, robot_name="atlas_v4_description", init_frame=30, frames_cut_end=55, nb_turns_in_vid=4, transformation_values = [np.pi, 0., -np.pi/2, 0., 0., 1., 2., 2., 2.]):

    def create_video(animation_frames, audio_file, background_image, output_file, credits_text):
        # Load background image
        background = cv2.imread(background_image)
        height, width, _ = background.shape

        # Calculate the frame rate to match the video length to the music length
        frame_rate = len(animation_frames) / mus_lenght

        # Create a video writer with the calculated frame rate
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter('temp_video.mp4', fourcc, frame_rate, (width, height))
        
        # Write animation frames to video
        for frame in animation_frames:
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
    mus_lenght = music_lenght(audio_file)

    # Animate the robot dancing
    animate_robot_dancing(movement_file, robot_name, bpm, init_frame, frames_cut_end, nb_turns_in_vid, transformation_values, mus_lenght)

    # Load animation frames
    animation_frames = []
    for i in range(len(os.listdir('frames/'))) :
        frame = cv2.imread(f'frames/frame_{i:04d}.png')
        if frame is None:
            break
        animation_frames.append(frame)

    # Create a video with the robot dancing to the music
    create_video(animation_frames, audio_file, background_image, output_file, credits_text)


if __name__ == '__main__':
    music_file_path = 'Chostakovitch_Kitaenko_w2.mp3'
    record_video_path = 'robot_waltz.mp4'
    background_image_path = 'ballroom.jpg'
    transformation_values = [np.pi, 0., -np.pi/2, 0., 0., 1., 2., 2., 2.]
    movement_file = 'valse_constantin.pt'

    credits_text = 'A video realized by Constantin Vaillant-Tenzer and Charles Monte' + '\n' \
                   + 'Music: Chostakovitch, waltz #2 - Directed by D. Kitaenko' + '\n' \

    # Example usage
    create_video_robot(movement_file, music_file_path, background_image_path, record_video_path, credits_text)
