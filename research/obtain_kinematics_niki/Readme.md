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

- **Videos**: Place the video(s) you want to analyze in the appropriate folder (e.g., `input/videos`). If the algorithm requires multiple camera angles, duplicate the video or provide additional perspectives.
- **Calibration**: Populate the `input/calibration` folder with calibration images or files necessary for accurate positional mapping.

### 2. Run NIKI Analysis

Ensure you are in the main project directory, then open an interactive Python console:

```bash
ipython
```

Run the following commands to execute the pipeline:

```python
from NIKI import NIKI

# Step-by-step process
NIKI.pose_estimation()    # Estimate poses from video
NIKI.calibration()        # Calibrate cameras
NIKI.synchronization()    # Synchronize multi-camera data
NIKI.triangulation()      # Triangulate 3D positions
NIKI.filtering()          # Apply filters to refine data
NIKI.cinematic_positions() # Compute cinematic positions
```

### 3. Output

The processed cinematic position data will be saved in the `output` folder:
- **3D Position Data**: Check the `output/3d_positions` folder.
- **Cinematic Metrics**: Review detailed cinematic metrics in the `output/cinematic_metrics` file.


## Additional Notes

- Ensure that input data is of high quality for optimal results.
- Use synchronized videos for multi-camera setups to avoid inconsistencies.
- Verify the output files for completeness and accuracy.

## References

For detailed instructions and updates, visit the [NIKI GitHub Repository](https://github.com/Jeff-sjtu/NIKI).

