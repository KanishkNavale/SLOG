import cv2
import numpy as np
import torch

from slog.utils import convert_tensor_to_cv2
from slog.renderers import annotate_point_in_image

import torchvision.transforms as T


def create_grid(spatial_probs: torch.Tensor) -> torch.Tensor:

    us = torch.arange(0, spatial_probs.shape[0], 1, dtype=torch.float32, device=spatial_probs.device)
    vs = torch.arange(0, spatial_probs.shape[1], 1, dtype=torch.float32, device=spatial_probs.device)
    grid = torch.meshgrid(us, vs, indexing='ij')

    grid_stack = torch.stack(grid)

    return torch.vstack([spatial_probs.permute(2, 0, 1), grid_stack])


if __name__ == "__main__":

    image = torch.as_tensor(cv2.imread("thesis/thesis/images/cap/wild_a.jpg") / 255, dtype=torch.float32)
    grid_image = create_grid(image)

    augmented_image = T.RandomAffine(degrees=60, translate=(0.0, 0.1))(grid_image)

    I, J = 0, 0
    for i in range(grid_image.shape[-2]):
        for j in range(grid_image.shape[-1]):
            if torch.allclose(augmented_image[3:, ...].permute(1, 2, 0)[i, j], torch.as_tensor([500., 500.])):
                I, J = i, j
                break

    # annotate
    image = convert_tensor_to_cv2(grid_image.permute(1, 2, 0)[..., :3])
    image = annotate_point_in_image(image, 500, 500, (255, 0, 0))

    augmented_image = convert_tensor_to_cv2(augmented_image.permute(1, 2, 0)[..., :3])
    augmented_image = annotate_point_in_image(augmented_image, I, J, (0, 255, 0))

    cv2.imwrite("Augmented Image.png", np.hstack([image, augmented_image]))
