# About references

- Un papier d'imitation de danse traditionnelle japonaise
- Un papier plus récent, qui correspond à un spectacle dont on peut trouver la vidéo en ligne.


## Other relevant links

https://ieeexplore.ieee.org/abstract/document/6231630

https://github.com/ProjectsAI/ComparativeArtisticEvaluation/tree/main

https://www.youtube.com/watch?v=qZimIZihYM8

Pour l'acquisition des mouvements de danse : vous pourriez par exemple vous filmer et acquérir des trajectoires cibles en utilisant l'un des projets listés dans ce dépôt (https://github.com/zongmianli/Estimating-3D-Motion-Forces), et vous pourriez regarder ce papier (https://openaccess.thecvf.com/content_CVPR_2019/papers/Li_Estimating_3D_Motion_and_Forces_of_Person-Object_Interactions_From_Monocular_CVPR_2019_paper.pdf).

Pour les modèles d'humanoïdes, vous en trouverez une collection dans ce projet (https://github.com/robot-descriptions/robot_descriptions.py?tab=readme-ov-file#humanoids). Enfin, pour la cinématique inverse, vous pouvez l'implémenter vous-même comme lors de votre TP ou faire appel cette bibliothèque (https://github.com/stephane-caron/pink).


Neural inverse Kinematics: https://github.com/Jeff-sjtu/NIKI.
la page du papier de HybrIK c'est https://github.com/Jeff-sjtu/HybrIK : Hybrid Analytical neural inverse kinematics for body mesh recovery.

## Humanoids robots benchmark

## Humanoids

| Name                      | Robot            | Maker              | DOF | Format | Notes |
|---------------------------|------------------|--------------------|-----|--------|------------|
| atlas_drc_description     | Atlas DRC (v3)   | Boston Dynamics    | 30  | URDF   |
| atlas_v4_description      | Atlas v4         | Boston Dynamics    | 30  | URDF   |
| berkeley_humanoid_description | Berkeley Humanoid | Hybrid Robotics | 12  | URDF   | Not found loads_robots |
| draco3_description        | Draco3           | Apptronik          | 25  | URDF   |
| ergocub_description       | ergoCub          | IIT                | 57  | URDF   |
| g1_description            | G1               | UNITREE Robotics   | 37  | URDF   |
| g1_mj_description         | G1               | UNITREE Robotics   | 37  | MJCF   |
| h1_description            | H1               | UNITREE Robotics   | 25  | URDF   |
| h1_mj_description         | H1               | UNITREE Robotics   | 25  | MJCF   |
| icub_description          | iCub             | IIT                | 32  | URDF   |
| jaxon_description         | JAXON            | JSK                | 38  | URDF   |
| jvrc_description          | JVRC-1           | AIST               | 34  | URDF   |
| jvrc_mj_description       | JVRC-1           | AIST               | 34  | MJCF   |
| op3_mj_description        | OP3              | ROBOTIS            | 20  | MJCF   |
| r2_description            | Robonaut 2       | NASA JSC Robotics  | 56  | URDF   |
| romeo_description         | Romeo            | Aldebaran Robotics | 37  | URDF   |
| sigmaban_description      | SigmaBan         | Rhoban             | 20  | URDF   |
| talos_description         | TALOS            | PAL Robotics       | 32  | URDF   |
| talos_mj_description      | TALOS            | PAL Robotics       | 32  | MJCF   |
| valkyrie_description      | Valkyrie         | NASA JSC Robotics  | 59  | URDF   |
