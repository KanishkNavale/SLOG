from typing import Tuple, List, Dict

import torch
from kornia.geometry.linalg import relative_transformation

from slog.keypoint_nets.configurations.config import Loss


class KeypointNetLosses:
    def __init__(self, loss_config: Loss) -> None:
        self.name = 'Keypoint Network Losses'

        # Init. configuration
        self.reduction = loss_config.reduction
        self.config = loss_config

    @staticmethod
    def _compute_spatial_expectations(depth: torch.Tensor,
                                      mask: torch.Tensor,
                                      spatial_probs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        us = torch.arange(0, depth.shape[-2], 1, dtype=torch.float32, device=depth.device)
        vs = torch.arange(0, depth.shape[-1], 1, dtype=torch.float32, device=depth.device)
        grid = torch.meshgrid(us, vs, indexing='ij')

        number_of_keypoints = spatial_probs.shape[0]

        exp_u = torch.sum(spatial_probs * grid[0].unsqueeze(dim=0).tile(number_of_keypoints, 1, 1), dim=(-2, -1))
        exp_v = torch.sum(spatial_probs * grid[1].unsqueeze(dim=0).tile(number_of_keypoints, 1, 1), dim=(-2, -1))
        exp_d = torch.sum(spatial_probs * depth, dim=(-2, -1))
        exp_m = torch.sum(spatial_probs * mask, dim=(-2, -1))

        return exp_u, exp_v, (exp_d + 1e-12), exp_m

    @staticmethod
    def _batch_convert_image_to_camera_coordinates(u: torch.Tensor,
                                                   v: torch.Tensor,
                                                   d: torch.Tensor,
                                                   intrinsic: torch.Tensor) -> torch.Tensor:
        du = u * d
        dv = v * d
        uvd = torch.stack([du, dv, d]).permute(1, 0).unsqueeze(dim=-1)
        intrinsic = intrinsic.unsqueeze(dim=0).tile(u.shape[0], 1, 1)
        return torch.bmm(torch.linalg.inv(intrinsic), uvd)

    def _compute_multiview_consistency_loss(self,
                                            cam_coords_a: torch.Tensor,
                                            extrinsic_a: torch.Tensor,
                                            cam_coords_b: torch.Tensor,
                                            extrinsic_b: torch.Tensor) -> torch.Tensor:

        one_padded_cam_coords_a = torch.hstack([cam_coords_a.squeeze(dim=-1),
                                                torch.ones_like(cam_coords_a.squeeze(dim=-1)[:, :1])]).unsqueeze(dim=-1)
        one_padded_cam_coords_b = torch.hstack([cam_coords_b.squeeze(dim=-1),
                                                torch.ones_like(cam_coords_b.squeeze(dim=-1)[:, :1])]).unsqueeze(dim=-1)

        tiled_extrinsics_a = extrinsic_a.unsqueeze(dim=0).tile(cam_coords_a.shape[0], 1, 1)
        tiled_extrinsics_b = extrinsic_b.unsqueeze(dim=0).tile(cam_coords_b.shape[0], 1, 1)

        world_coords_a = torch.bmm(tiled_extrinsics_a, one_padded_cam_coords_a)
        world_coords_b = torch.bmm(tiled_extrinsics_b, one_padded_cam_coords_b)

        return torch.dist(world_coords_a, world_coords_b)

    @staticmethod
    def _margin_loss(matrix: torch.Tensor, margin: float) -> torch.Tensor:

        intra_distance = torch.cdist(matrix.squeeze(dim=-1), matrix.squeeze(dim=-1))

        upper_triangle_indices_x, upper_triangle_indices_y = torch.triu_indices(intra_distance.shape[0], intra_distance.shape[1], offset=1)
        upper_triangle_matrix = intra_distance[upper_triangle_indices_x, upper_triangle_indices_y]

        min_distance = torch.min(upper_triangle_matrix)
        hinge = torch.max(torch.zeros_like(min_distance), margin - min_distance)

        return torch.mean(hinge)

    def _compute_separation_loss(self,
                                 points_a: torch.Tensor,
                                 points_b: torch.Tensor) -> torch.Tensor:

        return 0.5 * (self._margin_loss(points_a, self.config.margin) + self._margin_loss(points_b, self.config.margin))

    def _compute_silhoutte_loss(self,
                                exp_mask_a: torch.Tensor,
                                exp_mask_b: torch.Tensor) -> torch.Tensor:
        return 0.5 * (torch.mean(-torch.log(exp_mask_a + 1e-12)) + torch.mean(-torch.log(exp_mask_b + 1e-12)))

    @staticmethod
    def _variance_expectation(exp_pixel: torch.Tensor,
                              spatial_probs: torch.Tensor) -> torch.Tensor:

        us = torch.arange(0, spatial_probs.shape[-2], 1, dtype=torch.float32, device=spatial_probs.device)
        vs = torch.arange(0, spatial_probs.shape[-1], 1, dtype=torch.float32, device=spatial_probs.device)
        grid = torch.meshgrid(us, vs, indexing='ij')

        spatial_grid = torch.cat([grid[0].unsqueeze(-1), grid[0].unsqueeze(-1)], dim=-1)

        tiled_spatial_grid = spatial_grid.unsqueeze(dim=0).tile(exp_pixel.shape[0], 1, 1, 1)
        tiled_exp_pixel = exp_pixel.view(exp_pixel.shape[0], 1, 1, exp_pixel.shape[1])

        distances_summed = (torch.linalg.norm(tiled_spatial_grid - tiled_exp_pixel, dim=-1))

        probability_weights = torch.sum(spatial_probs * distances_summed, dim=(-2, -1))

        return torch.mean(probability_weights)

    def _compute_variance_loss(self,
                               exp_uv_a: torch.Tensor,
                               spatial_exp_a: torch.Tensor,
                               exp_uv_b: torch.Tensor,
                               spatial_exp_b: torch.Tensor) -> torch.Tensor:
        return 0.5 * (self._variance_expectation(exp_uv_a, spatial_exp_a) + self._variance_expectation(exp_uv_b, spatial_exp_b))

    @ staticmethod
    def _geodesic_distance(rotation_matrix_a: torch.Tensor,
                           rotation_matrix_b: torch.Tensor) -> torch.Tensor:

        matrix = rotation_matrix_a @ rotation_matrix_b.T
        identity = torch.eye(matrix.shape[-1], dtype=matrix.dtype, device=matrix.device)

        return 0.5 * torch.linalg.norm(identity - matrix)

    def _compute_relative_pose_loss(self,
                                    cam_coords_a: torch.Tensor,
                                    extrinsic_a: torch.Tensor,
                                    cam_coords_b: torch.Tensor,
                                    extrinsic_b,
                                    noise_std: float = 0.1) -> torch.Tensor:

        # Groundtruth transformations
        trafo_a_to_b = relative_transformation(extrinsic_a, extrinsic_b)
        trafo_b_to_a = relative_transformation(extrinsic_b, extrinsic_a)

        rot_trafo_a_to_b = trafo_a_to_b[:3, :3]
        rot_trafo_b_to_a = trafo_b_to_a[:3, :3]

        # Adding noise for robustness
        cam_coords_a += torch.normal(0.0, noise_std, size=cam_coords_a.size(), device=cam_coords_a.device, dtype=cam_coords_a.dtype)
        cam_coords_b += torch.normal(0.0, noise_std, size=cam_coords_b.size(), device=cam_coords_b.device, dtype=cam_coords_b.dtype)

        # Cam coord centers
        cam_coord_center_a = torch.mean(cam_coords_a, dim=0)
        cam_coord_center_b = torch.mean(cam_coords_b, dim=0)

        # Zero center the coordinates
        centered_coords_a = cam_coords_a - cam_coord_center_a
        centered_coords_b = cam_coords_b - cam_coord_center_b

        # Covariance matrices
        cov_a = (centered_coords_a.T @ centered_coords_b)
        cov_b = (centered_coords_b.T @ centered_coords_a)

        # SVD decomposition
        ua, _, vt_a = torch.linalg.svd(cov_a)
        ub, _, vt_b = torch.linalg.svd(cov_b)

        # Handling reflection case
        identity_matrix_a = identity_matrix_b = torch.eye(3, dtype=cam_coords_a.dtype, device=cam_coords_a.device)
        identity_matrix_a[-1, -1] = torch.linalg.det(vt_a.T @ ua.T)
        normalizer_a = identity_matrix_a
        identity_matrix_b[-1, -1] = torch.linalg.det(vt_b.T @ ub.T)
        normalizer_b = identity_matrix_b

        # Computed predicted transformation
        predicted_rot_a_to_b = vt_a.T @ normalizer_a @ ua.T
        predicted_rot_b_to_a = vt_b.T @ normalizer_b @ ub.T

        # Compute loss
        rotation_loss = 0.5 * (self._geodesic_distance(rot_trafo_a_to_b, predicted_rot_a_to_b) + self._geodesic_distance(rot_trafo_b_to_a, predicted_rot_b_to_a))

        return rotation_loss

    def _reduce_list(self, list_of_tensors: List[torch.Tensor]) -> torch.Tensor:
        return torch.sum(torch.vstack(list_of_tensors)) if self.config.reduction == "sum" else torch.mean(torch.vstack(list_of_tensors))

    def _compute_weighted_losses(self,
                                 consistency_loss: List[torch.Tensor],
                                 pose_loss: List[torch.Tensor],
                                 separation_loss: List[torch.Tensor],
                                 sihoutte_loss: List[torch.Tensor]) -> torch.Tensor:

        loss = torch.hstack([self._reduce_list(consistency_loss),
                             self._reduce_list(pose_loss),
                             self._reduce_list(separation_loss),
                             self._reduce_list(sihoutte_loss)])

        return self.config.loss_ratios_as_tensor.type(loss.dtype).to(loss.device) * loss

    def __call__(self,
                 depths_a: torch.Tensor,
                 depths_b: torch.Tensor,
                 intrinsics_a: torch.Tensor,
                 intrinsics_b: torch.Tensor,
                 extrinsics_a: torch.Tensor,
                 extrinsics_b: torch.Tensor,
                 masks_a: torch.Tensor,
                 masks_b: torch.Tensor,
                 spatial_probs_a: torch.Tensor,
                 spatial_probs_b: torch.Tensor) -> Dict[str, torch.Tensor]:

        consistency_loss: List[torch.Tensor] = []
        separation_loss: List[torch.Tensor] = []
        # variance_loss: List[torch.Tensor] = []
        pose_loss: List[torch.Tensor] = []
        silhoutte_loss: List[torch.Tensor] = []

        for (depth_a,
             depth_b,
             intrinsic_a,
             intrinsic_b,
             extrinsic_a,
             extrinsic_b,
             mask_a,
             mask_b,
             spatial_prob_a,
             spatial_prob_b) in zip(depths_a,
                                    depths_b,
                                    intrinsics_a,
                                    intrinsics_b,
                                    extrinsics_a,
                                    extrinsics_b,
                                    masks_a,
                                    masks_b,
                                    spatial_probs_a,
                                    spatial_probs_b):

            exp_ua, exp_va, exp_da, exp_ma = self._compute_spatial_expectations(depth_a, mask_a, spatial_prob_a)
            exp_ub, exp_vb, exp_db, exp_mb = self._compute_spatial_expectations(depth_b, mask_b, spatial_prob_b)

            cam_coords_a = self._batch_convert_image_to_camera_coordinates(exp_ua, exp_va, exp_da, intrinsic_a)
            cam_coords_b = self._batch_convert_image_to_camera_coordinates(exp_ub, exp_vb, exp_db, intrinsic_b)

            consistency_loss.append(self._compute_multiview_consistency_loss(cam_coords_a,
                                                                             extrinsic_a,
                                                                             cam_coords_b,
                                                                             extrinsic_b))

            separation_loss.append(self._compute_separation_loss(torch.vstack([exp_ua, exp_va]).permute(1, 0),
                                                                 torch.vstack([exp_ub, exp_vb]).permute(1, 0)))

            silhoutte_loss.append(self._compute_silhoutte_loss(exp_ma, exp_mb))

            pose_loss.append(self._compute_relative_pose_loss(cam_coords_a.squeeze(dim=-1),
                                                              extrinsic_a,
                                                              cam_coords_b.squeeze(dim=-1),
                                                              extrinsic_b))

        weighted_batch_loss = self._compute_weighted_losses(consistency_loss,
                                                            pose_loss,
                                                            separation_loss,
                                                            silhoutte_loss)

        return {"Total": torch.sum(weighted_batch_loss),
                "Consistency": weighted_batch_loss[0],
                "Relative Pose": weighted_batch_loss[1],
                "Separation": weighted_batch_loss[2],
                "Silhoutte": weighted_batch_loss[3],
                # "Variance": weighted_batch_loss[4],
                }
