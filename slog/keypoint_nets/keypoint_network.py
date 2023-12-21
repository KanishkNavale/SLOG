import numpy as np
import torch
import torch.nn.functional as F
import pytorch_lightning as pl

import slog.don.__models as models
from slog.keypoint_nets import KeypointNetConfig
from slog.keypoint_nets.losses import KeypointNetLosses
from slog.utils import initialize_config_file


class KeypointNetwork(pl.LightningModule):

    def __init__(self, yaml_config_path: str) -> None:
        super(KeypointNetwork, self).__init__()

        # Init. configuration
        config_dictionary = initialize_config_file(yaml_config_path)
        self.config = KeypointNetConfig.from_dictionary(config_dictionary)
        self.yaml_config_path = yaml_config_path

        # Init. parent model
        if self.config.keypointnet.backbone == "resnet_18":
            self.resnet = models.resnet18(fully_conv=True,
                                          pretrained=True,
                                          output_stride=8,
                                          remove_avg_pool_layer=True)
        elif self.config.keypointnet.backbone == "resnet_34":
            self.resnet = models.resnet34(fully_conv=True,
                                          pretrained=True,
                                          output_stride=8,
                                          remove_avg_pool_layer=True)
        else:
            self.resnet = models.resnet50(fully_conv=True,
                                          pretrained=True,
                                          output_stride=8,
                                          remove_avg_pool_layer=True)

        self.resnet.avgpool = torch.nn.Identity()
        self.resnet.fc = torch.nn.Conv2d(self.resnet.inplanes, self.config.keypointnet.bottleneck_dimension, 1, bias=False)

        self.spatial_layer = torch.nn.Conv2d(self.config.keypointnet.bottleneck_dimension,
                                             self.config.keypointnet.n_keypoints,
                                             kernel_size=3,
                                             padding='same')

        self.depth_layer = torch.nn.Conv2d(self.config.keypointnet.bottleneck_dimension,
                                           self.config.keypointnet.n_keypoints,
                                           kernel_size=3,
                                           padding='same')

        # Init. Loss
        self.loss_function = KeypointNetLosses(self.config.loss)

        # random colors for debugging
        self.colors = [(np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)) for _ in range(self.config.keypointnet.n_keypoints)]

    def _forward(self, input: torch.Tensor) -> torch.Tensor:
        x = self.resnet.forward(input)
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

    def configure_optimizers(self):
        if self.config.optimizer.name == 'Adam':
            optimizer = torch.optim.Adam(self.parameters(),
                                         lr=self.config.optimizer.learning_rate)
        else:
            raise NotImplementedError(f"This optimizer: {self.config.optimizer.name} is not implemented")

        return optimizer

    def training_step(self, batch, batch_idx):

        batch_pcls_a = batch["RGBs-A"]
        batch_pcls_b = batch["RGBs-B"]

        spat_exp_a, depth_a = self._forward(batch_pcls_a)
        spat_exp_b, depth_b = self._forward(batch_pcls_b)

        batch_loss = self.loss_function(depth_a,
                                        depth_b,
                                        batch["Intrinsics-A"],
                                        batch["Intrinsics-B"],
                                        batch["Extrinsics-A"],
                                        batch["Extrinsics-B"],
                                        batch["Masks-A"],
                                        batch["Masks-B"],
                                        spat_exp_a,
                                        spat_exp_b)

        self.log("Training Loss", {**batch_loss})

        return batch_loss["Total"]

    def validation_step(self, batch, batch_idx):

        batch_pcls_a = batch["RGBs-A"]
        batch_pcls_b = batch["RGBs-B"]

        spat_exp_a, depth_a = self._forward(batch_pcls_a)
        spat_exp_b, depth_b = self._forward(batch_pcls_b)

        batch_loss = self.loss_function(depth_a,
                                        depth_b,
                                        batch["Intrinsics-A"],
                                        batch["Intrinsics-B"],
                                        batch["Extrinsics-A"],
                                        batch["Extrinsics-B"],
                                        batch["Masks-A"],
                                        batch["Masks-B"],
                                        spat_exp_a,
                                        spat_exp_b)

        self.log("val_pass", {**batch_loss})

        return batch_loss["Total"]

    def validation_epoch_end(self, outputs):
        loss = torch.sum(torch.hstack(outputs)) if self.config.loss.reduction == 'sum' else torch.mean(torch.hstack(outputs))
        self.log("Validation Loss", loss)
        return loss

    def get_dense_representation(self, input: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            x = self.resnet.forward(input)
            return F.interpolate(x,
                                 size=input.size()[-2:],
                                 mode='bilinear',
                                 align_corners=True)
