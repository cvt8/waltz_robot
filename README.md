# Make an humanoid robot dance the waltz
## Project Description

Authors: Charles Monte and Constantin Vaillant-Tenzer

This project simulates and realizes the inverse kinematics of a robot dancing waltz. We primarily focused on using the Atlas v4 robot from Boston Dynamics. However, our code can be easily adapted to work with any robot by making the necessary semantic adjustments.

## How to Run the Code

To generate the video from our code, follow these steps:

1. Clone the project repository: `$ git clone https://github.com/cvt8/waltz_robot.git`
2. Navigate to the repository: `$ cd waltz_robot` (for Linux users)
3. Create a new Python environment: `$ conda env create -f environment.yml`
4. Activate the conda environment `$ conda activate waltz_robot`
5. Run the code: `$ python waltz_robot.py`
6. The generated video will be saved as `waltz_video.mp4` in the project directory.

## Repository Organization

The repository is organized into the following directories:

- `src`: Contains the source code for implementing the inverse kinematics and generating a video synchronized with music.
- `research`: Includes the research materials we used to generate files and make informed decisions.
- `poster`: Contains the LaTeX files for creating the project poster.
- `report`: Includes the LaTeX files for the project report.

## Steps - Example Usage

### Make a Video of Yourself Dancing

To make a video of yourself dancing the waltz, you can follow these steps:

1. Set up a camera to record yourself dancing.
2. Choose a suitable location with enough space to move around.
3. Put on your dancing shoes and start dancing the waltz.
4. Record your dance moves using the camera.
5. Save the video file on your computer.

For intance, we started from this video:
![](/research/obtain_kinematics_niki/VID20241126155236.mp4) 


### Use Niki to Generate a New Video

To generate a new video using Niki, follow these steps:

1. Clone the Niki repository from [here](https://github.com/Jeff-sjtu/NIKI).
2. Install the required dependencies as mentioned in the repository's README.
3. Run the Niki script with the appropriate parameters to generate a new video.
4. Save the generated video file on your computer.

You will get a video like this:
![](/research/obtain_kinematics_niki/res_valse_constantin.mp4) 


### Select a Music

Choose a suitable music track to accompany your waltz video. Consider selecting a waltz music piece to match the dance style. In our exemple, we use the Waltz#2 of Chostokovitch.

### Generate Your Implementation and Your Video

To generate your implementation and video, follow these steps:

1. Write your code to implement the inverse kinematics of the robot dancing waltz.
2. Test and debug your code to ensure it works correctly.
3. Run your code to generate the robot's dance moves.
4. Combine the robot's dance moves with the selected music track to create a synchronized video.
5. Save the final video file on your computer.

## How to Contribute

If you would like to contribute to this project, please follow these guidelines:

1. Fork the repository.
2. Create a new branch for your contributions.
3. Make your changes and improvements.
4. Test your changes to ensure they do not introduce any issues.
5. Submit a pull request with a detailed description of your changes.

## How to Cite Our Work

If you use our work in your research or project, please cite it using the following LaTeX citation:

```
@article{monte_vaillant_tenzer_2024,
    title={Inverse Kinematics of a Robot Dancing Waltz},
    author={Monte, Charles and Vaillant-Tenzer, Constantin},
    journal={},
    year={2024},
    volume={X},
    number={X},
    pages={X-X},
    doi={X.XXX/XXXXXX}
}
```

## Report Link

For a detailed report on our project, please refer to [insert link to the project report].

## YouTube Video Link

To watch a video demonstration of our robot dancing waltz, please visit [insert link to YouTube video].