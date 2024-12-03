#!/bin/bash

# Author; Constantin Vaillant-Tenzer
# Date: 2021-07-01
# This script is used to show the URDF of the humanoids in MeshCat.
# Execution: bash show_in_meshcat_humanoids.sh
# The option --joints_only or -j will show only the joints of the robot.


robots_to_test=(
    "atlas_drc_description" "atlas_v4_description" "draco3_description"
    "ergocub_description" "g1_description" "g1_mj_description" "h1_description" "h1_mj_description"
    "icub_description" "jaxon_description" "jvrc_description" "jvrc_mj_description" "op3_mj_description"
    "r2_description" "romeo_description" "sigmaban_description" "talos_description" "talos_mj_description"
    "valkyrie_description"
)

# Show only the joints of the robot
if [[ $1 == "--joints_only" ]] || [[ $1 == "-j" ]]; then
    for robot in "${robots_to_test[@]}"; do
        echo "Showing $robot in MeshCat with only the joints"
        python show_joints_robots.py "$robot"
    done

else 
    for robot in "${robots_to_test[@]}"; do
    echo "Showing $robot in MeshCat"
    python show_in_robomeshcat.py "$robot"
done
fi