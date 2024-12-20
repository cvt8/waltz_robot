import os
import subprocess
import numpy as np
import cv2
#from src.robot_waltz_from_niki import animate_robot_dancing
from src.bpm_detection import get_bpm
from src.music_lenght_detection import music_lenght


def create_video_robot(audio_file, background_image, output_file, credits_text, robot_name="atlas_v4_description", init_frame=30, frames_cut_end=55, nb_turns_in_vid=4):

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
    
    bpm = get_bpm(music_file_path)

    animation_frames = animate_robot_dancing(robot_name, bpm, init_frame, frames_cut_end, nb_turns_in_vid)
    
    # Create a video with the robot dancing to the music
    create_video(animation_frames, audio_file, background_image, output_file, credits_text)


if __name__ == '__main__':
    music_file_path = 'Chostakovitch_Kitaenko_w2.mp3'
    record_video_path = 'robot_waltz.mp4'
    background_image_path = 'ballroom.jpg'

    credits_text = 'A video realized by Constantin Vaillant-Tenzer and Charles Monte' + '\n' \
                   + 'Music: Chostakovitch, waltz #2 - Directed by D. Kitaenko' + '\n' \

    music_lenght = music_lenght(music_file_path)
    print(f"Lenght: {music_lenght}")

    # Example usage
    create_video(music_file_path, background_image_path, record_video_path, credit_text)
