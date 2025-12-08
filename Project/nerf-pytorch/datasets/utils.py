import numpy as np


def load_pose_vecs(poses_bound_path):
    poses_bound = np.load(poses_bound_path)

    poses_flat = poses_bound[:,:-2]
    poses = poses_flat.reshape(-1,3,5)

    pose_vecs = []

    for i in range(len(poses)):
        pose3x5 = poses[i] # [3,5]
        c2w_3x4 = pose3x5[:,:4] # [3,4]
        pose_vec = c2w_3x4.reshape(-1).astype(np.float32) # 12D
        pose_vecs.append(pose_vec)

    return np.stack(pose_vecs, axis=0) # [N, 12]