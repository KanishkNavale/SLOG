from typing import List, Tuple

import torch
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as T
import pytorch_lightning as pl

import slog.don.__models as models
from slog.don import DenseObjectNetConfig, PixelwiseCorrespondenceLoss, PixelwiseNTXentLoss
from slog.utils import initialize_config_file


class DON(pl.LightningModule):

    def __init__(self, yaml_config_path: str) -> None:
        super(DON, self).__init__()

        # Init. configuration
        config_dictionary = initialize_config_file(yaml_config_path)
        self.config = DenseObjectNetConfig.from_dictionary(config_dictionary)

        # Init. Backbone
        if self.config.don.backbone == "resnet_18":
            self.model = models.resnet18(fully_conv=True,
                                         pretrained=True,
                                         output_stride=8,
                                         remove_avg_pool_layer=True)
        elif self.config.don.backbone == "resnet_34":
            self.model = models.resnet34(fully_conv=True,
                                         pretrained=True,
                                         output_stride=8,
                                         remove_avg_pool_layer=True)
        else:
            self.model = models.resnet50(fully_conv=True,
                                         pretrained=True,
                                         output_stride=8,
                                         remove_avg_pool_layer=True)

        self.model.avgpool = torch.nn.Identity()
        self.model.fc = torch.nn.Conv2d(self.model.inplanes, self.config.don.descriptor_dimension, 1, bias=False)

        # Init. loss function
        if self.config.loss.name == 'pixelwise_correspondence_loss':
            self.loss_function = PixelwiseCorrespondenceLoss(reduction=self.config.loss.reduction)

        elif self.config.loss.name == 'pixelwise_ntxent_loss':
            self.loss_function = PixelwiseNTXentLoss(self.config.loss.temperature, self.config.loss.reduction)

        # Init. data augmentation
        self.input_augmentor = T.Compose([T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    def configure_optimizers(self):

        if self.config.optimizer.name == 'Adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=self.config.optimizer.learning_rate, weight_decay=1e-4)

        else:
            raise NotImplementedError(f"This optimizer: {self.config.optimizer.name} is not implemented")

        if self.config.optimizer.enable_schedular:
            sch = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.config.optimizer.schedular_step_size, gamma=0.99)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": sch,
                    "monitor": "train_loss",

                }
            }
        else:
            return optimizer

    def _forward(self, input: torch.Tensor) -> torch.Tensor:
        feature_map = self.model(self.input_augmentor(input))
        scaled_map = F.interpolate(feature_map,
                                   size=input.size()[2:],
                                   mode='bilinear',
                                   align_corners=True)
        return scaled_map

    @staticmethod
    def _stack_image_with_mask_and_grid(images: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        images = images.permute(0, 3, 1, 2)
        masks = masks.unsqueeze(dim=1)

        us = torch.arange(0, images.shape[-2], 1, dtype=torch.float32, device=images.device)
        vs = torch.arange(0, images.shape[-1], 1, dtype=torch.float32, device=images.device)
        grid = torch.meshgrid(us, vs, indexing='ij')
        spatial_grid = torch.stack(grid)

        tiled_spatial_grid = spatial_grid.unsqueeze(dim=0).tile(images.shape[0], 1, 1, 1)

        return torch.cat([images, masks, tiled_spatial_grid], dim=1)

    @staticmethod
    def _destack_image_mask_spatialgrid(augmented_image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        image = augmented_image[:, :3, :, :]
        mask = augmented_image[:, 3, :, :]
        spatial_grid = torch.round(augmented_image[:, 4:, :])

        return image, mask, spatial_grid

    @staticmethod
    def _get_random_augmentation(image: torch.Tensor) -> torch.Tensor:
        if 1 == np.random.randint(0, 3):
            return T.RandomAffine(degrees=60, translate=(0, 0.2))(image)

        elif 1 == np.random.randint(0, 3):
            return T.RandomPerspective(distortion_scale=0.2, p=1.0)(image)

        elif 1 == np.random.randint(0, 3):
            return T.RandomVerticalFlip(p=1.0)(image)

        else:
            return image

    def _compute_correspondence_and_augmented_images(self,
                                                     images_a: torch.Tensor,
                                                     images_b: torch.Tensor,
                                                     masks: torch.Tensor,) -> torch.Tensor:

        augmented_image_a = self._get_random_augmentation(self._stack_image_with_mask_and_grid(images_a, masks))
        augmented_image_b = self._get_random_augmentation(self._stack_image_with_mask_and_grid(images_b, masks))

        augmented_images_a, masks_a, grids_a = self._destack_image_mask_spatialgrid(augmented_image_a)
        augmented_images_b, _, grids_b = self._destack_image_mask_spatialgrid(augmented_image_b)

        matches_a: List[torch.Tensor] = []
        matches_b: List[torch.Tensor] = []

        for mask_a, grid_a, grid_b in zip(masks_a, grids_a, grids_b):

            valid_pixels_a = torch.where(mask_a != 0.0)

            us = valid_pixels_a[0]
            vs = valid_pixels_a[1]

            # Reducing computation costs
            trimming_indices = torch.linspace(0, us.shape[0] - 1, steps=10 * self.config.datamodule.n_correspondence)
            us = us[trimming_indices.long()].type(torch.float32)
            vs = vs[trimming_indices.long()].type(torch.float32)

            valid_pixels_a = torch.vstack([us, vs]).permute(1, 0)
            valid_grids_a = grid_a[:, us.long(), vs.long()].permute(1, 0)
            tiled_valid_grids_a = valid_grids_a.view(valid_grids_a.shape[0], valid_grids_a.shape[1], 1, 1)

            spatial_grid_distances = torch.linalg.norm(grid_b - tiled_valid_grids_a, dim=1)

            match_indices_a, ubs, vbs = torch.where(spatial_grid_distances == 0.0)

            mutual_match_a = valid_pixels_a[match_indices_a.long()]
            mutual_match_b = torch.vstack([ubs, vbs]).permute(1, 0)

            trimming_indices = torch.linspace(0, mutual_match_a.shape[0] - 1, steps=self.config.datamodule.n_correspondence)
            trimming_indices = trimming_indices.type(torch.int64)

            matches_a.append(mutual_match_a[trimming_indices])
            matches_b.append(mutual_match_b[trimming_indices])

        return augmented_images_a, torch.stack(matches_a), augmented_images_b, torch.stack(matches_b)

    def training_step(self, batch, batch_idx):

        image_a, image_b, mask = batch["Anchor-Image"], batch["Augmented-Image"], batch["Mask"]

        image_a, matches_a, image_b, matches_b = self._compute_correspondence_and_augmented_images(image_a, image_b, mask)

        dense_descriptors_a, dense_descriptors_b = self._forward(image_a), self._forward(image_b)

        loss = self.loss_function(dense_descriptors_a, dense_descriptors_b, matches_a, matches_b)

        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):

        image_a, image_b, mask = batch["Anchor-Image"], batch["Augmented-Image"], batch["Mask"]

        image_a, matches_a, image_b, matches_b = self._compute_correspondence_and_augmented_images(image_a, image_b, mask)

        dense_descriptors_a, dense_descriptors_b = self._forward(image_a), self._forward(image_b)

        loss = self.loss_function(dense_descriptors_a, dense_descriptors_b, matches_a, matches_b)

        self.log("val_loss", loss)

        return loss

    def compute_dense_features(self, input: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self._forward(input)
