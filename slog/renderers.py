import cv2
import numpy as np

import torch

from slog import utils
from slog.datatypes import Pose


def render_spatial_distribution(image: torch.Tensor, spatial_probability: torch.Tensor) -> np.ndarray:
    """Superposes spatial heatmap on to a image.

    Args:
        image (torch.Tensor): Any Image of dimension (H x W X D)
        spatial_probability (torch.Tensor): Spatial probability of dimension (H x W)

    Returns:
        np.ndarray: cv2 format of heatmap superimposed image
    """

    spatial_probability = spatial_probability.unsqueeze(dim=-1).tile(1, 1, 3)
    image = image[:, :, :3]

    scaled_image = utils.convert_tensor_to_cv2(image)
    spatial_probability = utils.convert_tensor_to_cv2(spatial_probability)

    heatmap = cv2.applyColorMap(spatial_probability, cv2.COLORMAP_JET)
    superposed_image = cv2.addWeighted(scaled_image, 0.5, heatmap, 0.7, 0)

    return superposed_image


def render_pose(img: np.ndarray, pose: Pose, intrinsic: torch.Tensor, axis_lenght: float = 0.5) -> np.ndarray:
    """Renders the pose axes on the image.

    Args:
        img (np.ndarray): Image of cv2(uint8) datatype
        pose (Pose): Pose datatype
        intrinsic (torch.Tensor): camera intrinsic parameter
        axis_lenght (float, optional): axis lenght (units of intrinsic). Defaults to 0.05.

    Returns:
        np.ndarray: rotation axes annotated image of cv2 uint8 format
    """

    intrinsic = utils.convert_tensor_to_numpy(intrinsic)
    rotation = utils.convert_tensor_to_numpy(pose.rotation)

    center = pose.translation_as_numpy
    rotation, _ = cv2.Rodrigues(rotation)

    dist = np.zeros(4, dtype=center.dtype)

    # Handle distortion
    points = axis_lenght * np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=dist.dtype)

    axis_points, _ = cv2.projectPoints(points, rotation, center, intrinsic, dist)
    axis_points = axis_points.astype(np.int64)

    img = cv2.line(img, tuple(axis_points[3].ravel()), tuple(axis_points[0].ravel()), (255, 0, 0), 3)
    img = cv2.line(img, tuple(axis_points[3].ravel()), tuple(axis_points[1].ravel()), (0, 255, 0), 3)
    img = cv2.line(img, tuple(axis_points[3].ravel()), tuple(axis_points[2].ravel()), (0, 0, 255), 3)

    return img


def annotate_point_in_image(image: np.ndarray, u: int, v: int, color: np.ndarray, add_text: bool = True) -> np.ndarray:
    font = cv2.FONT_HERSHEY_COMPLEX
    cv2.circle(image, (int(v), int(u)), radius=1, color=color, thickness=2)
    if add_text:
        image = cv2.putText(image, f'{u, v}', (v + 5, u - 5), font, 0.3, color, 1, cv2.LINE_AA)
    return image
