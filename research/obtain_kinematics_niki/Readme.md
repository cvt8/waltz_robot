# Obtaining Cinematic Positions Using NIKI

This guide explains how to use the NIKI framework to obtain cinematic positions, inspired by similar workflows like obtaining kinematic data through Pose2Sim.

## Overview

NIKI is a tool designed for motion analysis and visualization. This README provides instructions to reproduce the cinematic position results using NIKI.

## Installation

Clone the [NIKI repository](https://github.com/Jeff-sjtu/NIKI):

```bash
git clone https://github.com/Jeff-sjtu/NIKI.git
cd NIKI
```

Follow the installation instructions provided in the repository to set up the environment. Ensure you have Python and the required dependencies installed.

## Workflow

### 1. Prepare Input Data

- **Videos**: Place the video(s) you want to analyze in the appropriate folder (e.g., `input/videos`).
- **Pretrained models**: Download the pretrained HybrIK and Single-stage NIKI models from [this link](https://sjtueducn-my.sharepoint.com/:f:/g/personal/biansiyuan_sjtu_edu_cn/EtGnxMf0bkpPhB8OPecnzhoBbKzgXrhyVguV_B5g4r8_rQ?e=UGDdRJ), and put them in the exp/ folder.

### 2. Run NIKI Analysis

Ensure you are in the main project directory, then run the following command:

```bash
python scripts/demo.py --video-name {VIDEO-PATH} -out-dir {OUTPUT-DIR}
```

Replace `{VIDEO-PATH}` with the path to the video file you want to analyze and `{OUTPUT-DIR}` with the directory where you want to save the output files.

### 3. Output

The processed cinematic position data will be saved in the `{OUTPUT-DIR}` folder you specified. The output includes:
- **raw_images**: In the `raw_images` folder, you will find the raw images extracted from the video.
- **res_images**: The `res_images` folder contains the images after processing, where the optimal humanoid model is overlaid on the raw images.
- **res_{VIDEO-NAME}.mp4**: The processed video with the humanoid model overlaid on the original video.
- **{VIDEO-NAME}.pt**: The data extracted from the video, which is used by our model to generate the robot animation. It can be opened using Python's `joblib` library for instance and notably contains the 3D positions of the joints in the referential of the pelvis (`pred_xyz_29` key) and the position of the pelvis in the referential of the camera (`pred_cam_root` key).


## Additional Notes

- High quality videos are not mandatory and can significantly increase processing time. However, a stable video with a clear view of the whole body is mandatory.
- Verify the output files for completeness and accuracy.

## References

For detailed instructions and updates, visit the [NIKI GitHub Repository](https://github.com/Jeff-sjtu/NIKI).

