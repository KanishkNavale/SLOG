from typing import List, Tuple

import torch

from slog.distances import l2

"""
NOTE: This loss function is implemented from,

Dense Visual Learning for Robot Manipulation - Peter R. Florence - Ph.D Thesis - MIT

Chapter 4: Dense Spatial-Temporal Distributional Correspondence Learning

"""


# @torch.jit.script
def _compute_correspondence_loss(grid: Tuple[torch.Tensor, torch.Tensor],
                                 image_a: torch.Tensor,
                                 image_b: torch.Tensor,
                                 matches_a: torch.Tensor,
                                 matches_b: torch.Tensor) -> torch.Tensor:
    queried_descriptors_a = image_a[:, matches_a[:, 0].long(), matches_a[:, 1].long()].permute(1, 0)
    tiled_queried_desciptors_a = queried_descriptors_a.reshape((matches_a.shape[0], 1, 1, image_a.shape[0]))
    tiled_image_b = image_b.unsqueeze(0).tile(matches_a.shape[0], 1, 1, 1).permute(0, 2, 3, 1)

    spatial_distances_of_descriptors_a_in_image_b = l2(tiled_image_b, tiled_queried_desciptors_a, dim=-1)
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

    return torch.nn.functional.huber_loss(spatial_expectation_uv, matches_b)


class PixelwiseCorrespondenceLoss:
    def __init__(self, reduction: str) -> None:
        self.name = 'Pixelwise Correspondence Loss'
        self.reduction = reduction

    def _compute_directional_wise_loss(self,
                                       batch_image_a: torch.Tensor,
                                       batch_image_b: torch.Tensor,
                                       batch_matches_a: torch.Tensor,
                                       batch_matches_b: torch.Tensor) -> torch.Tensor:

        us = torch.arange(0, batch_image_a.shape[-2], 1, dtype=torch.float32, device=batch_image_a.device)
        vs = torch.arange(0, batch_image_a.shape[-1], 1, dtype=torch.float32, device=batch_image_a.device)
        grid = torch.meshgrid(us, vs, indexing='ij')

        batch_loss: List[torch.Tensor] = []
        for image_a, image_b, matches_a, matches_b in zip(batch_image_a, batch_image_b, batch_matches_a, batch_matches_b):
            batch_loss.append(_compute_correspondence_loss(grid, image_a, image_b, matches_a, matches_b))

        # convert list -> tensor
        batch_loss = torch.hstack(batch_loss)

        if self.reduction == "sum":
            return torch.sum(batch_loss)

        elif self.reduction == "mean":
            return torch.mean(batch_loss)

    def __call__(self,
                 batch_image_a: torch.Tensor,
                 batch_image_b: torch.Tensor,
                 batch_matches_a: torch.Tensor,
                 batch_matches_b: torch.Tensor) -> torch.Tensor:

        backward_loss = self._compute_directional_wise_loss(batch_image_a,
                                                            batch_image_b,
                                                            batch_matches_a,
                                                            batch_matches_b)

        forward_loss = self._compute_directional_wise_loss(batch_image_b,
                                                           batch_image_a,
                                                           batch_matches_b,
                                                           batch_matches_a)

        return 0.5 * (backward_loss + forward_loss)
