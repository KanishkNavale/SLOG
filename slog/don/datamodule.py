from typing import List, Dict, Any
import os

import numpy as np
import cv2
from sklearn.model_selection import train_test_split
import random

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import pytorch_lightning as pl

from slog.don import DenseObjectNetConfig
from slog.utils import initialize_config_file

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


class DonDataset(Dataset):
    def __init__(self,
                 rgbs: List[str],
                 masks: List[str],
                 config: DenseObjectNetConfig) -> None:

        self.rgbs = rgbs
        self.masks = masks

        self.config = config
        self.random_backgrounds = [os.path.join("dataset/random_backgrounds", file) for file in os.listdir("dataset/random_backgrounds")]

    def __len__(self) -> int:
        return len(self.rgbs)

    def _augment_image(self,
                       image: torch.Tensor,
                       mask: torch.Tensor) -> torch.Tensor:

        output = image

        # Random Background
        if 1 == np.random.randint(0, 5):
            tiled_mask = mask.unsqueeze(dim=-1).tile(1, 1, 3)
            masked_image = image * tiled_mask

            random_image = cv2.imread(random.choice(self.random_backgrounds)) / 255
            random_image = torch.as_tensor(cv2.resize(random_image, (image.shape[1], image.shape[0])), dtype=image.dtype)

            masked_random_image = torch.where(masked_image != torch.zeros(3, dtype=masked_image.dtype),
                                              torch.zeros(3, dtype=masked_image.dtype),
                                              random_image)

            output = masked_image + masked_random_image

            return output

        # Gaussian Blur
        elif 1 == np.random.randint(0, 5):
            blurred_image: torch.Tensor = T.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5))(image.permute(2, 0, 1))
            return blurred_image.permute(1, 2, 0)

        # Greyscale augmentation
        elif 1 == np.random.randint(0, 5):
            grayscale: torch.Tensor = T.Grayscale()(image.permute(2, 0, 1))
            output = grayscale.tile(3, 1, 1).permute(1, 2, 0)

        # Noisy background
        elif 1 == np.random.randint(0, 3):
            tiled_mask = mask.unsqueeze(dim=-1).tile(1, 1, 3)
            masked_image = image * tiled_mask

            random_image = torch.rand_like(masked_image)
            masked_random_image = torch.where(masked_image != torch.zeros(3, dtype=masked_image.dtype),
                                              torch.zeros(3, dtype=masked_image.dtype),
                                              random_image)

            output = masked_random_image + masked_image

        # Masked image
        elif 1 == np.random.randint(0, 5):
            tiled_mask = mask.unsqueeze(dim=-1).tile(1, 1, 3)
            masked_image = image * tiled_mask
            output = masked_image

        return output

    def __getitem__(self, idx: int) -> Dict[str, Any]:

        rgb = self.rgbs[idx]
        mask = self.masks[idx]

        rgb = torch.as_tensor(cv2.imread(rgb) / 255, dtype=torch.float32)
        mask = torch.as_tensor(cv2.imread(mask) / 255, dtype=torch.float32)[..., 0]

        return {
            "Anchor-Image": rgb,
            "Augmented-Image": self._augment_image(rgb, mask),
            "Mask": mask

        }


class DataModuleDON(pl.LightningDataModule):
    def __init__(self, yaml_config_path: str) -> None:

        config_dictionary = initialize_config_file(yaml_config_path)
        config = DenseObjectNetConfig.from_dictionary(config_dictionary)
        self.config = config

        # Default values
        self._log_hyperparams = self.config.datamodule.n_workers
        self.prepare_data_per_node = True

    def prepare_data(self) -> None:
        # Reading RGBD data
        rgb_directory = self.config.datamodule.rgb_directory
        masks_directory = self.config.datamodule.mask_directory

        self.rgb_files = sorted([os.path.join(rgb_directory, file) for file in os.listdir(rgb_directory)])
        self.mask_files = sorted([os.path.join(masks_directory, file) for file in os.listdir(masks_directory)])

    def setup(self, stage: str = None):
        # Create training, validation datasplits
        (train_rgbs,
         val_rgbs,
         train_masks,
         val_masks) = train_test_split(self.rgb_files,
                                       self.mask_files,
                                       shuffle=self.config.datamodule.shuffle,
                                       test_size=self.config.datamodule.test_size)

        if stage == 'fit':
            self.training_dataset = DonDataset(train_rgbs,
                                               train_masks,
                                               self.config)

            self.validation_dataset = DonDataset(val_rgbs,
                                                 val_masks,
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
