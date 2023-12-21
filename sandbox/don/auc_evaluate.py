from typing import List, Tuple, Dict

import os

import numpy as np
import cv2
import torch
import random

from slog.don import DON
from slog.inference import PCK
from slog.geometry import invert_extrinsic_matrix

from tqdm import tqdm

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def check_if_pixel_is_valid(pixel: torch.Tensor, mask: torch.Tensor) -> bool:
    u = pixel[0]
    v = pixel[1]
    return torch.allclose(mask[u, v], torch.ones(1, dtype=mask.dtype, device=mask.device))


def compute_correspondences(
        depth_a: torch.Tensor,
        mask_a: torch.Tensor,
        extrinsic_a: torch.Tensor,
        intrinsic_a: torch.Tensor,
        mask_b: torch.Tensor,
        extrinsic_b: torch.Tensor,
        intrinsic_b: torch.Tensor,
        n_correspondence: int) -> Tuple[torch.Tensor, torch.Tensor]:

    # Collect all pixels inside mask of image_a
    valid_pixels_a = torch.where(mask_a == 1.0)
    list_world_coords_image_a: List[torch.Tensor] = []

    for u, v in zip(valid_pixels_a[0], valid_pixels_a[1]):
        depth = depth_a[u, v] + 1e-8
        cam_coords = torch.linalg.inv(intrinsic_a) @ (depth * torch.vstack([u.to(device), v.to(device), torch.ones_like(depth)]))
        world_coords = (extrinsic_a) @ torch.vstack([cam_coords, torch.ones_like(depth)])
        list_world_coords_image_a.append(world_coords[:3].type(torch.float32).reshape((1, 3)))

    # Project points in camera b
    list_valid_uvs_b: List[torch.Tensor] = []
    list_valid_uvs_a: List[torch.Tensor] = []

    for coord, ua, va in zip(list_world_coords_image_a, valid_pixels_a[0], valid_pixels_a[1]):
        cam_coord_b = invert_extrinsic_matrix(extrinsic_b) @ torch.vstack([coord.permute(1, 0),
                                                                           torch.ones(1, device=coord.device, dtype=coord.dtype)])
        duvb = intrinsic_b @ cam_coord_b[:3]

        depth = duvb[-1]
        uvb = torch.divide(duvb[:2], depth).type(torch.int64)

        pixel_a = torch.hstack([ua, va])
        pixel_b = uvb.reshape(pixel_a.shape)

        if check_if_pixel_is_valid(pixel_b, mask_b):
            list_valid_uvs_a.append(pixel_a)
            list_valid_uvs_b.append(pixel_b)

    matches_a = torch.vstack(list_valid_uvs_a)
    matches_b = torch.vstack(list_valid_uvs_b)

    trimming_indices = torch.as_tensor(random.sample(range(0, len(matches_a)), n_correspondence))
    trimming_indices = trimming_indices.type(torch.int64)

    matches_a = matches_a[trimming_indices].type(torch.float32)
    matches_b = matches_b[trimming_indices].type(torch.float32)

    return matches_a, matches_b


if __name__ == "__main__":

    model = DON("sandbox/don/don-config.yaml")
    model: DON = model.load_from_checkpoint(
        checkpoint_path='/home/kanishk/sereact/slog_models/don/don-d3-resnet34-nc1024-correspondence.ckpt',
        yaml_config_path="sandbox/don/don-config.yaml"
    )
    model.to(device)

    rgb_directory = "dataset/single_object_dataset/dataset/rgbs"
    depth_directory = "dataset/single_object_dataset/dataset/depths"
    masks_directory = "dataset/single_object_dataset/dataset/masks"
    extrinsic_directory = "dataset/single_object_dataset/dataset/extrinsics_cam_to_world"
    intrinsic_matrix = torch.as_tensor(np.loadtxt("dataset/single_object_dataset/dataset/intrinsics.txt"), dtype=torch.float32, device=device)

    rgb_files = sorted([os.path.join(rgb_directory, file) for file in os.listdir(rgb_directory)])
    depth_files = sorted([os.path.join(depth_directory, file) for file in os.listdir(depth_directory)])
    mask_files = sorted([os.path.join(masks_directory, file) for file in os.listdir(masks_directory)])
    extrinsic_files = sorted([os.path.join(extrinsic_directory, file) for file in os.listdir(extrinsic_directory)])

    AUCS = []
    for j in tqdm(range(3), desc="Benchmarking AUC of PCK@k", total=3):
        AUC = []
        for k in tqdm(range(1, 100), desc=f"Iteration ~ {j}", total=99):
            random_index = np.random.randint(low=0, high=len(rgb_files) - 1)

            rgb_a = rgb_files[random_index]
            depth_a = depth_files[random_index]
            mask_a = mask_files[random_index]
            extrinsic_a = extrinsic_files[random_index]
            intrinsic_a = intrinsic_matrix

            rgb_a = torch.as_tensor(cv2.imread(rgb_a) / 255, dtype=torch.float32, device=device)
            depth_a = torch.as_tensor(np.load(depth_a, allow_pickle=False), dtype=torch.float32, device=device)
            mask_a = torch.as_tensor(cv2.imread(mask_a) / 255, dtype=torch.float32)[..., 0]
            extrinsic_a = torch.as_tensor(np.loadtxt(extrinsic_a), dtype=torch.float32, device=device)
            intrinsic_a = torch.as_tensor(intrinsic_a, dtype=torch.float32, device=device)

            descriptor_a = model.compute_dense_features(rgb_a.permute(2, 0, 1).unsqueeze(0)).squeeze(dim=0).to(device)

            random_index = np.random.randint(low=0, high=len(rgb_files) - 1)

            rgb_b = rgb_files[random_index]
            depth_b = depth_files[random_index]
            mask_b = mask_files[random_index]
            extrinsic_b = extrinsic_files[random_index]
            intrinsic_b = intrinsic_matrix

            rgb_b = torch.as_tensor(cv2.imread(rgb_b) / 255, dtype=torch.float32, device=device)
            depth_b = torch.as_tensor(np.load(depth_b, allow_pickle=False), dtype=torch.float32, device=device)
            mask_b = torch.as_tensor(cv2.imread(mask_b) / 255, dtype=torch.float32, device=device)[..., 0]
            extrinsic_b = torch.as_tensor(np.loadtxt(extrinsic_b), dtype=torch.float32, device=device)
            intrinsic_b = torch.as_tensor(intrinsic_b, dtype=torch.float32, device=device)

            descriptor_b = model.compute_dense_features(rgb_b.permute(2, 0, 1).unsqueeze(0)).squeeze(dim=0).to(device)

            matches_a, matches_b = compute_correspondences(depth_a,
                                                           mask_a,
                                                           extrinsic_a,
                                                           intrinsic_a,
                                                           mask_b,
                                                           extrinsic_b,
                                                           intrinsic_b,
                                                           100)

            pck = PCK(descriptor_image_a=descriptor_a,
                      descriptor_image_b=descriptor_b,
                      matches_a=matches_a,
                      matches_b=matches_b,
                      k=k)

            AUC.append(pck)
        print(AUC)
        AUC = torch.trapz(torch.hstack(AUC), dx=1.0)
        AUCS.append(AUC)

    AUCS = torch.hstack(AUCS)
    print(f"AUC:{AUCS.mean()}, sigma={AUCS.std()}")
