from geometry import *

def axis_angle_to(newtype, rotations):
    if newtype in ["matrix"]:
        rotations = axis_angle_to_matrix(rotations)
        return rotations
    elif newtype in ["rotmat"]:
        rotations = axis_angle_to_matrix(rotations)
        rotations = matrix_to("rotmat", rotations)
        return rotations
    elif newtype in ["rot6d", "6drot", "rotation6d"]:
        rotations = axis_angle_to_matrix(rotations)
        rotations = matrix_to("rot6d", rotations)
        return rotations
    elif newtype in ["rotquat", "quaternion"]:
        rotations = axis_angle_to_quaternion(rotations)
        return rotations
    elif newtype in ["rotvec", "axisangle"]:
        return rotations
    else:
        raise NotImplementedError


def matrix_to(newtype, rotations):
    if newtype in ["matrix"]:
        return rotations
    if newtype in ["rotmat"]:
        rotations = rotations.reshape((*rotations.shape[:-2], 9))
        return rotations
    elif newtype in ["rot6d", "6drot", "rotation6d"]:
        rotations = matrix_to_rotation_6d(rotations)
        return rotations
    elif newtype in ["rotquat", "quaternion"]:
        rotations = matrix_to_quaternion(rotations)
        return rotations
    elif newtype in ["rotvec", "axisangle"]:
        rotations = matrix_to_axis_angle(rotations)
        return rotations
    else:
        raise NotImplementedError


def to_matrix(oldtype, rotations):
    if oldtype in ["matrix"]:
        return rotations
    if oldtype in ["rotmat"]:
        rotations = rotations.reshape((*rotations.shape[:-2], 3, 3))
        return rotations
    elif oldtype in ["rot6d", "6drot", "rotation6d"]:
        rotations = rotation_6d_to_matrix(rotations)
        return rotations
    elif oldtype in ["rotquat", "quaternion"]:
        rotations = quaternion_to_matrix(rotations)
        return rotations
    elif oldtype in ["rotvec", "axisangle"]:
        rotations = axis_angle_to_matrix(rotations)
        return rotations
    else:
        raise NotImplementedError


def inverse(trans_feature):
    root_y = trans_feature[..., 0]
    vel_trajectory = trans_feature[..., 1:3]

    # integrate the trajectory
    trajectory = torch.cumsum(vel_trajectory.squeeze(-2), dim=-2)
    # First frame should be 0, but if infered it is better to ensure it
    trajectory = trajectory - trajectory[..., [0], :]

    # Get back the translation
    trans = torch.cat([trajectory.unsqueeze(2), root_y[..., None]], dim=-1)

    return trans
