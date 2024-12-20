import numpy as np
from scipy.optimize import minimize

def apply_transformation(transformation_matrix, positions):
    """Apply transformation matrix to the positions.
    
    Args:
        transformation_matrix: Transformation matrix, shape (4, 4)
        positions: Positions to be transformed, shape (t, n, 3)
    """
    if positions.ndim == 2:
        positions_bigger = np.concatenate((positions, np.ones((positions.shape[0], 1))), axis=1)
        transformed_positions_bigger = (transformation_matrix@positions_bigger.T).T
        transformed_positions = transformed_positions_bigger[:, :3]
        
    else:
        transformed_positions = np.zeros_like(positions)
        for t in range(positions.shape[0]):
            actual_position = positions[t]
            actual_position_bigger = np.concatenate((actual_position, np.ones((actual_position.shape[0], 1))), axis=1)
            transformed_position_bigger = (transformation_matrix@actual_position_bigger.T).T
            transformed_positions[t] = transformed_position_bigger[:, :3]
            
    return transformed_positions

def get_transformation_matrix(values):
    """Get the transformation matrix from the parameters.
    
    Args:
        values: List of 9 values [alpha, beta, gamma, x, y, z, sx, sy, sz]
        alpha, beta, gamma: Rotation angles around x, y, z axes
        x, y, z: Translation along x, y, z axes
        sx, sy, sz: Scaling along x, y, z axes
        
    Returns:
        trans_mat: Transformation matrix
    """
    rot_angles = values[:3]; translation = values[3:6]; scale = values[6:9]
    alpha = rot_angles[0]; beta = rot_angles[1]; gamma = rot_angles[2]
    ca = np.cos(alpha); sa = np.sin(alpha)
    cb = np.cos(beta); sb = np.sin(beta)
    cg = np.cos(gamma); sg = np.sin(gamma)
    rot_mat = np.array([[ca*cb, ca*sb*sg - sa*cg, ca*sb*cg + sa*sg],
                        [sa*cb, sa*sb*sg + ca*cg, sa*sb*cg - ca*sg],
                        [-sb, cb*sg, cb*cg]])
    rot_mat[:3, 0] *= scale[0]
    rot_mat[:3, 1] *= scale[1]
    rot_mat[:3, 2] *= scale[2]
    trans_mat = np.eye(4)
    trans_mat[:3, :3] = rot_mat
    trans_mat[:3, 3] = translation
    return trans_mat

def find_optimal_transformation_matrix(positions, init_frame, max_frame, element_markers):
    """Find the optimal transformation matrix to align the movement with the (x, y) plane.
    
    Args:
        positions: Positions, shape (t, n, 3)
        init_frame: Initial frame
        max_frame: Maximum frame
        element_markers: Element markers
        
    Returns
    """

    # Getting the optimal transformation matrix to align the movement with the (x, y) plane
    head_pose = positions[int(init_frame):int(max_frame), element_markers["head"]]
    pelvis_pose = positions[int(init_frame):int(max_frame), element_markers["pelvis"]]
    r_foot_pose = positions[int(init_frame):int(max_frame), element_markers["r_foot"]]
    l_foot_pose = positions[int(init_frame):int(max_frame), element_markers["l_foot"]]

    def cost(transformation_parameters):
        trans_mat = get_transformation_matrix(transformation_parameters)
        transformed_head = apply_transformation(transformation_matrix=trans_mat, positions=head_pose)
        transformed_pelvis = apply_transformation(transformation_matrix=trans_mat, positions=pelvis_pose)
        transformed_r_foot = apply_transformation(transformation_matrix=trans_mat, positions=r_foot_pose)
        transformed_l_foot = apply_transformation(transformation_matrix=trans_mat, positions=l_foot_pose)

        cost = np.sum(transformed_head[:, 2] - 1.5)**2
        cost += np.sum(transformed_pelvis[:, 0] - transformed_head[:, 0])**2
        cost += np.sum(transformed_pelvis[:, 1] - transformed_head[:, 1])**2
        cost += np.abs(np.sum(transformed_l_foot[:, 2]))
        cost += np.abs(np.sum(transformed_r_foot[:, 2]))
        return cost
    
    start_values = [0, 0, 0, 0, 0, 0, 1, 1, 1]
    res = minimize(cost, start_values, method="Powell")
    transformation_matrix = get_transformation_matrix(res.x)
    
    return transformation_matrix
    
    