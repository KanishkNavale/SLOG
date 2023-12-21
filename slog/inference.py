import torch

from slog import distances


def compute_keypoint_expectation(image: torch.Tensor,
                                 keypoint: torch.Tensor,
                                 temp: torch.Tensor) -> torch.Tensor:
    weights = distances.exp_guassian_distance_kernel(image, keypoint, temp)
    spatial_probabilities = weights / weights.sum()
    sum_of_spatial_probs = torch.sum(spatial_probabilities)

    if not torch.allclose(sum_of_spatial_probs, torch.ones_like(sum_of_spatial_probs)):
        raise ValueError("Spatial probabilities don't add upto 1.0")
    else:
        return spatial_probabilities


def compute_keypoint_confident_expectation(image: torch.Tensor,
                                           keypoint: torch.Tensor,
                                           temp: torch.Tensor,
                                           confidence: torch.Tensor) -> torch.Tensor:
    spatial_expectation = compute_keypoint_expectation(image, keypoint, temp)
    normalized_expectation = spatial_expectation / spatial_expectation.max()
    zero = torch.zeros(1, device=normalized_expectation.device, dtype=normalized_expectation.dtype)
    return torch.where(normalized_expectation >= confidence, normalized_expectation, zero)


def is_rotationmatrix(rotation_matrix: torch.Tensor, tolerance: float = 1e-3) -> bool:
    """Checks if the matrix suffice the rotational matrix properties

    Args:
        rotation_matrix (torch.Tensor): matrix of shape (3 x 3)

    Returns:
        bool: is rotation matrix ?
    """
    R = rotation_matrix.clone()
    E = torch.eye(3, device=R.device, dtype=rotation_matrix.dtype)
    determinant_R = torch.linalg.det(R)

    return torch.allclose(R.T @ R, E, atol=tolerance) and torch.allclose(determinant_R, torch.ones_like(determinant_R), atol=tolerance)


def PCK(descriptor_image_a: torch.Tensor,
        descriptor_image_b: torch.Tensor,
        matches_a: torch.Tensor,
        matches_b: torch.Tensor,
        k: float) -> torch.Tensor:

    k = (k + 1) / 100

    us = torch.arange(0, descriptor_image_a.shape[-2], 1, dtype=torch.float32, device=descriptor_image_a.device)
    vs = torch.arange(0, descriptor_image_a.shape[-1], 1, dtype=torch.float32, device=descriptor_image_a.device)
    grid = torch.meshgrid(us, vs, indexing='ij')

    queried_descriptors_a = descriptor_image_a[:, matches_a[:, 0].long(), matches_a[:, 1].long()].permute(1, 0)
    tiled_queried_desciptors_a = queried_descriptors_a.reshape((matches_a.shape[0], 1, 1, descriptor_image_a.shape[0]))
    tiled_image_b = descriptor_image_b.unsqueeze(0).tile(matches_a.shape[0], 1, 1, 1).permute(0, 2, 3, 1)

    spatial_distances_of_descriptors_a_in_image_b = distances.l2(tiled_image_b, tiled_queried_desciptors_a, dim=-1)
    kernel_distances = torch.exp(-1.0 * torch.square(spatial_distances_of_descriptors_a_in_image_b)) + 1e-16
    spatial_probabilities = kernel_distances / torch.sum(kernel_distances, dim=(1, 2), keepdim=True)

    if not torch.allclose(torch.sum(spatial_probabilities, dim=(1, 2)),
                          torch.ones(matches_a.shape[0],
                                     dtype=spatial_probabilities.dtype,
                                     device=spatial_probabilities.device)):
        raise ValueError("Spatial probabilities do not add up to 1.0")

    spatial_expectations_u = torch.sum(torch.multiply(grid[0], spatial_probabilities), dim=(1, 2))
    spatial_expectations_v = torch.sum(torch.multiply(grid[1], spatial_probabilities), dim=(1, 2))
    spatial_expectation_uv = torch.hstack([spatial_expectations_u[:, None], spatial_expectations_v[:, None]])

    indices_similarity = distances.cosine_similarity(matches_b, spatial_expectation_uv)

    iversion_operation = indices_similarity >= k

    inversion = iversion_operation.type(torch.float32)

    return inversion.mean()
