from typing import List, Dict, Any
import os

import numpy as np
import random
import cv2
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.functional import adjust_gamma
import pytorch_lightning as pl

from slog.keypoint_nets.configurations.config import KeypointNetConfig
from slog.utils import initialize_config_file

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


class KeypointNetDataset(Dataset):
    def __init__(self,
                 rgbs: List[torch.Tensor],
                 depths: List[torch.Tensor],
                 masks: List[torch.Tensor],
                 extrinsics: List[torch.Tensor],
                 intrinsic: torch.Tensor,
                 config: KeypointNetConfig) -> None:
        self.rgbs = rgbs
        self.depths = depths
        self.masks = masks
        self.extrinsics = extrinsics
        self.intrinsics = [intrinsic] * self.__len__()
        self.config = config

    def __len__(self) -> int:
        return len(self.depths)

    @ staticmethod
    def _add_illumination_augmentation(input: torch.Tensor) -> torch.Tensor:
        if torch.allclose(torch.randint(low=0, high=10, size=(1,), dtype=torch.float32), torch.ones((1,), dtype=torch.float32)):
            return adjust_gamma(input.permute(2, 0, 1), np.random.uniform(low=0.1, high=1.5)).permute(1, 2, 0)
        else:
            return input

    @ staticmethod
    def _add_background_randomization(input: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if torch.allclose(torch.randint(low=0, high=10, size=(1,), dtype=torch.float32), torch.ones((1,), dtype=torch.float32)):
            for u, v in zip(*torch.where(mask == 0.0)):
                input[u, v] = torch.rand(np.random.choice([1, 3]), dtype=input.dtype, device=input.device)
            return input
        else:
            return input

    @staticmethod
    def _add_mask(input: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        tiled_mask = mask.unsqueeze(dim=-1).tile(1, 1, input.shape[-1])
        return input * tiled_mask

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rgb_a = self.rgbs[idx]
        depth_a = self.depths[idx]
        mask_a = self.masks[idx]
        extrinsic_a = self.extrinsics[idx]
        intrinsic_a = self.intrinsics[idx]

        next_idx = random.randint(0, self.__len__() - 1)

        rgb_b = self.rgbs[next_idx]
        depth_b = self.depths[next_idx]
        mask_b = self.masks[next_idx]
        extrinsic_b = self.extrinsics[next_idx]
        intrinsic_b = self.intrinsics[next_idx]

        rgb_a = torch.as_tensor(cv2.imread(rgb_a) / 255, dtype=torch.float32)
        depth_a = torch.as_tensor(np.load(depth_a, allow_pickle=False) / self.config.datamodule.depth_ratio, dtype=torch.float32)
        mask_a = torch.as_tensor(cv2.imread(mask_a) / 255, dtype=torch.float32)[..., 0]
        extrinsic_a = torch.as_tensor(np.loadtxt(extrinsic_a), dtype=torch.float32)
        intrinsic_a = torch.as_tensor(intrinsic_a, dtype=torch.float32)

        rgb_b = torch.as_tensor(cv2.imread(rgb_b) / 255, dtype=torch.float32)
        depth_b = torch.as_tensor(np.load(depth_b, allow_pickle=False) / self.config.datamodule.depth_ratio, dtype=torch.float32)
        mask_b = torch.as_tensor(cv2.imread(mask_b) / 255, dtype=torch.float32)[..., 0]
        extrinsic_b = torch.as_tensor(np.loadtxt(extrinsic_b), dtype=torch.float32)
        intrinsic_b = torch.as_tensor(intrinsic_b, dtype=torch.float32)

        if self.config.datamodule.background_randomization:
            rgb_a = self._add_background_randomization(rgb_a, mask_a)
            rgb_b = self._add_background_randomization(rgb_b, mask_b)

        if self.config.datamodule.random_illumination:
            rgb_a = self._add_illumination_augmentation(rgb_a)
            rgb_b = self._add_illumination_augmentation(rgb_b)

        if self.config.datamodule.masked_training:
            rgb_a = self._add_mask(rgb_a, mask_a)
            rgb_b = self._add_mask(rgb_b, mask_b)
            depth_a = self._add_mask(depth_a.unsqueeze(dim=-1), mask_a).squeeze(dim=-1)
            depth_b = self._add_mask(depth_b.unsqueeze(dim=-1), mask_b).squeeze(dim=-1)

        return {
            "RGBs-A": rgb_a.permute(2, 0, 1),
            "RGBs-B": rgb_b.permute(2, 0, 1),
            "Depths-A": depth_a,
            "Depths-B": depth_b,
            "Intrinsics-A": intrinsic_a,
            "Intrinsics-B": intrinsic_b,
            "Extrinsics-A": extrinsic_a,
            "Extrinsics-B": extrinsic_b,
            "Masks-A": mask_a,
            "Masks-B": mask_b
        }


class DataModuleKeypointNet(pl.LightningDataModule):
    def __init__(self, yaml_config_path: str) -> None:

        config_dictionary = initialize_config_file(yaml_config_path)
        config = KeypointNetConfig.from_dictionary(config_dictionary)
        self.config = config

        # Default values
        self._log_hyperparams = self.config.datamodule.n_workers
        self.prepare_data_per_node = True

    def prepare_data(self) -> None:
        # Reading RGBD data
        rgb_directory = self.config.datamodule.rgb_directory
        depth_directory = self.config.datamodule.depth_directory
        masks_directory = self.config.datamodule.mask_directory
        extrinsic_directory = self.config.datamodule.extrinsic_directory
        self.intrinsic_matrix = np.loadtxt(self.config.datamodule.camera_intrinsics_numpy_text)

        self.rgb_files = sorted([os.path.join(rgb_directory, file) for file in os.listdir(rgb_directory)])
        self.depth_files = sorted([os.path.join(depth_directory, file) for file in os.listdir(depth_directory)])
        self.mask_files = sorted([os.path.join(masks_directory, file) for file in os.listdir(masks_directory)])
        self.extrinsic_files = sorted([os.path.join(extrinsic_directory, file) for file in os.listdir(extrinsic_directory)])

    def setup(self, stage: str = None):
        # Create training, validation datasplits
        (train_rgbs,
         val_rgbs,
         train_depths,
         val_depths,
         train_masks,
         val_masks,
         train_extrinsics,
         val_extrinsics) = train_test_split(self.rgb_files,
                                            self.depth_files,
                                            self.mask_files,
                                            self.extrinsic_files,
                                            shuffle=self.config.datamodule.shuffle,
                                            test_size=self.config.datamodule.test_size)

        if stage == 'fit':
            self.training_dataset = KeypointNetDataset(train_rgbs,
                                                       train_depths,
                                                       train_masks,
                                                       train_extrinsics,
                                                       self.intrinsic_matrix,
                                                       self.config)

            self.validation_dataset = KeypointNetDataset(val_rgbs,
                                                         val_depths,
                                                         val_masks,
                                                         val_extrinsics,
                                                         self.intrinsic_matrix,
                                                         self.config)

    def train_dataloader(self):
        return DataLoader(self.training_dataset,
                          num_workers=self.config.datamodule.n_workers,
                          batch_size=self.config.datamodule.batch_size,
                          pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.validation_dataset,
                          num_workers=self.config.datamodule.n_workers,
                          batch_size=self.config.datamodule.batch_size,
                          pin_memory=True)
