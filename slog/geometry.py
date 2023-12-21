import torch

from slog.inference import is_rotationmatrix
from slog.datatypes import Pose


def compute_pose(coordinates: torch.Tensor) -> Pose:
    """Computes rigid body transformation matrix from a tensor of 3D-Points

    Args:
        coordinates (torch.Tensor): Tensor of 3D-Points of dimension (N x 3)

    Returns:
        Pose: Pose datatype
    """

    if not coordinates.shape[-1] == 3:
        raise ValueError("Input tensor's last dimension is not 3")

    translation = torch.mean(coordinates, dim=0)
    mean_centering = coordinates - translation
    covariance = (mean_centering.T @ mean_centering) / mean_centering.shape[0]

    _, vectors = torch.linalg.eig(covariance)
    R = torch.real(vectors)

    rotation = torch.vstack((R[0], R[1], torch.cross(R[0], R[1])))

    if not is_rotationmatrix(rotation):
        raise ValueError("The computed rotation matrix doesn't suffice the rotational matrix properties")

    return Pose.from_translation_and_rotation(translation, rotation)


def compute_relative_pose(source: torch.Tensor, target: torch.Tensor) -> Pose:
    """Computes target's pose relative to the source

    Args:
        source (torch.Tensor): Tensor of 3D-Points of dimension (N x 3)
        target (torch.Tensor): Tensor of 3D-Points of dimension (N x 3)

    Returns:
        Pose: Pose datatype
    """
    source_pose = compute_pose(source)
    target_pose = compute_pose(target)

    relative_pose = torch.linalg.lstsq(target_pose.rigid_body_transformation, source_pose.rigid_body_transformation)[0]

    return Pose.from_rigid_body_transformation(relative_pose)


def invert_extrinsic_matrix(extrinsic_matrix: torch.Tensor) -> torch.Tensor:
    rotation = extrinsic_matrix[:3, :3]
    translation = extrinsic_matrix[:3, -1]

    stack = torch.hstack([rotation.T, -1.0 * rotation.T @ translation[:, None]])
    padding = torch.as_tensor([0.0, 0.0, 0.0, 1.0], dtype=rotation.dtype, device=rotation.device)

    return torch.vstack([stack, padding])
