from typing import Union

import torch
import torch.nn.functional as F
import pytorch_lightning as pl

from slog.don.__models import resnet18, resnet34, resnet50
from slog.don import PixelwiseCorrespondenceLoss, PixelwiseNTXentLoss
from slog.keypoint_nets import KeypointNetLosses
from slog.ende_zu_ende.don_informed_keypointnet import DONInformedKeypointNetConfig
from slog.utils import initialize_config_file


class DONInformedKeypointNet(pl.LightningModule):

    def __init__(self, yaml_config_path: str) -> None:
        super(DONInformedKeypointNet, self).__init__()

        # Init. configuration
        config_dictionary = initialize_config_file(yaml_config_path)
        self.config = DONInformedKeypointNetConfig.from_dictionary(config_dictionary)
        self.yaml_config_path = yaml_config_path

        # Init. KeypointNet
        self.keypointnet = self._fetch_resnet(self.config.keypointnet.backbone)
        self.keypointnet.fc = torch.nn.Conv2d(self.keypointnet.inplanes,
                                              self.config.keypointnet.bottleneck_dimension,
                                              1,
                                              bias=False)
        self.keypointnet.conv1 = torch.nn.Conv2d(self.config.don.descriptor_dimension,
                                                 64,
                                                 kernel_size=7,
                                                 stride=2,
                                                 padding=3,
                                                 bias=False)

        self.spatial_layer = torch.nn.Conv2d(self.config.keypointnet.bottleneck_dimension,
                                             self.config.keypointnet.n_keypoints,
                                             kernel_size=3,
                                             padding='same')

        self.depth_layer = torch.nn.Conv2d(self.config.keypointnet.bottleneck_dimension,
                                           self.config.keypointnet.n_keypoints,
                                           kernel_size=3,
                                           padding='same')

        # Init. DON
        self.don = self._fetch_resnet(self.config.don.backbone)
        self.don.fc = torch.nn.Conv2d(self.don.inplanes, self.config.don.descriptor_dimension, 1, bias=False)

        # Init. Losses
        self.keypointnet_loss = KeypointNetLosses(loss_config=self.config.loss)

        if self.config.loss.don == 'pixelwise_correspondence_loss':
            self.don_loss = PixelwiseCorrespondenceLoss(reduction=self.config.loss.reduction)

        elif self.config.loss.don == 'pixelwise_ntxent_loss':
            self.don_loss = PixelwiseNTXentLoss(self.config.loss.temperature, self.config.loss.reduction)

    @staticmethod
    def _fetch_resnet(name: str) -> Union[resnet18, resnet34, resnet50]:
        if name == "resnet_50":
            model = resnet50(fully_conv=True,
                             pretrained=True,
                             output_stride=8,
                             remove_avg_pool_layer=True)
        elif name == "resnet_34":
            model = resnet34(fully_conv=True,
                             pretrained=True,
                             output_stride=8,
                             remove_avg_pool_layer=True)
        else:
            model = resnet18(fully_conv=True,
                             pretrained=True,
                             output_stride=8,
                             remove_avg_pool_layer=True)

        model.avgpool = torch.nn.Identity()

        return model

    def _forward_keypointnet(self, input: torch.Tensor) -> torch.Tensor:
        x = self.keypointnet.forward(input)
        resized_x = F.interpolate(x,
                                  size=input.size()[-2:],
                                  mode='bilinear',
                                  align_corners=True)

        spatial_weights = self.spatial_layer.forward(resized_x)
        flat_weights = spatial_weights.reshape(spatial_weights.shape[0],
                                               spatial_weights.shape[1],
                                               spatial_weights.shape[2] * spatial_weights.shape[3])

        flat_probs = F.softmax(flat_weights, dim=-1)
        spatial_probs = flat_probs.reshape(flat_probs.shape[0],
                                           flat_probs.shape[1],
                                           input.shape[2],
                                           input.shape[3])

        depth = self.depth_layer.forward(resized_x)

        return spatial_probs, depth

    def _forward_don(self, input: torch.Tensor) -> torch.Tensor:
        feature_map = self.don(input)
        scaled_map = F.interpolate(feature_map,
                                   size=input.size()[2:],
                                   mode='bilinear',
                                   align_corners=True)
        return scaled_map

    def configure_optimizers(self):
        if self.config.optimizer.name == 'Adam':
            optimizer = torch.optim.Adam(self.parameters(),
                                         lr=self.config.optimizer.learning_rate)
        else:
            raise NotImplementedError(f"This optimizer: {self.config.optimizer.name} is not implemented")

        return optimizer

    def _shared_step(self, batch) -> torch.Tensor:

        batch_rgb_a = batch["RGBs-A"]
        batch_rgb_b = batch["RGBs-B"]

        batch_don_a = self._forward_don(batch_rgb_a)
        batch_don_b = self._forward_don(batch_rgb_b)

        batch_loss_don = self.don_loss(batch_don_a,
                                       batch_don_b,
                                       batch["Matches-A"],
                                       batch["Matches-B"])

        if self.current_epoch >= self.config.keypointnet.start_keypointnet_on_epoch:

            spat_exp_a, depth_a = self._forward_keypointnet(batch_don_a)
            spat_exp_b, depth_b = self._forward_keypointnet(batch_don_b)

            batch_loss_keypointnet = self.keypointnet_loss(depth_a,
                                                           depth_b,
                                                           batch["Intrinsics-A"],
                                                           batch["Intrinsics-B"],
                                                           batch["Extrinsics-A"],
                                                           batch["Extrinsics-B"],
                                                           batch["Masks-A"],
                                                           batch["Masks-B"],
                                                           spat_exp_a,
                                                           spat_exp_b)

            loss = batch_loss_keypointnet["Total"] + batch_loss_don

        else:
            loss = batch_loss_don

        return loss

    def training_step(self, batch, batch_idx):

        loss = self._shared_step(batch)
        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):

        loss = self._shared_step(batch)
        self.log("val_loss", loss)

        return loss
