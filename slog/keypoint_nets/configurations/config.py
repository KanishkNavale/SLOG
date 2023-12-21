from __future__ import annotations
from typing import Dict, Any
from dataclasses import dataclass
import os

import torch


@dataclass
class KeypointNet:
    n_keypoints: int
    bottleneck_dimension: int
    backbone: str

    @classmethod
    def from_dictionary(cls, dictionary: Dict[str, Any]) -> KeypointNet:
        return cls(**dictionary)

    def __post_init__(self):
        if self.backbone not in ["resnet_18", "resnet_34", "resnet_50"]:
            raise NotImplementedError(f"The specified backbone function: {self.backbone} is not implemented")


@dataclass
class Optimizer:
    name: str
    learning_rate: float
    enable_schedular: bool
    schedular_step_size: float

    @classmethod
    def from_dictionary(cls, dictionary: Dict[str, Any]) -> Optimizer:
        return cls(**dictionary)


@dataclass
class Loss:
    multiview_consistency: float
    relative_pose: float
    separation: float
    silhouette: float
    # variance: float

    reduction: str
    margin: float

    @property
    def loss_ratios_as_tensor(self) -> torch.Tensor:
        return torch.as_tensor([self.multiview_consistency,
                                self.relative_pose,
                                self.separation,
                                self.silhouette])
        # self.variance])

    @classmethod
    def from_dictionary(cls, dictionary: Dict[str, Any]) -> Loss:
        return cls(**dictionary)


@dataclass
class Trainer:
    epochs: int
    enable_logging: bool
    tensorboard_path: bool
    training_directory: str
    enable_checkpointing: bool
    model_path: str
    logging_frequency: bool
    validation_frequency: int

    def __post_init__(self):
        self.training_directory = os.path.abspath(self.training_directory)
        self.model_path = os.path.abspath(self.model_path)
        self.tensorboard_path = os.path.abspath(self.tensorboard_path)

    @classmethod
    def from_dictionary(cls, dictionary: Dict[str, Any]) -> Trainer:
        return cls(**dictionary)


@dataclass
class DataModule:
    rgb_directory: str
    depth_directory: str
    mask_directory: str
    extrinsic_directory: str
    camera_intrinsics_numpy_text: str

    random_illumination: bool
    background_randomization: bool
    masked_training: bool

    depth_ratio: str
    test_size: float
    shuffle: int

    n_workers: int
    batch_size: int

    @classmethod
    def from_dictionary(cls, dictionary: Dict[str, Any]) -> Trainer:
        return cls(**dictionary)

    def __post_init__(self):
        self.rgb_directory = os.path.abspath(self.rgb_directory)
        self.depth_directory = os.path.abspath(self.depth_directory)
        self.mask_directory = os.path.abspath(self.mask_directory)
        self.extrinsic_directory = os.path.abspath(self.extrinsic_directory)
        self.camera_intrinsics_numpy_text = os.path.abspath(self.camera_intrinsics_numpy_text)


@dataclass
class KeypointNetConfig:

    keypointnet: KeypointNet
    optimizer: Optimizer
    loss: Loss
    trainer: Trainer
    datamodule: DataModule

    @classmethod
    def from_dictionary(cls, dictionary: Dict[str, Any]) -> KeypointNetConfig:
        keypointnet = KeypointNet.from_dictionary(dictionary.keypointnet)
        optimizer = Optimizer.from_dictionary(dictionary.optimizer)
        loss = Loss.from_dictionary(dictionary.loss)
        trainer = Trainer.from_dictionary(dictionary.trainer)
        datamodule = DataModule.from_dictionary(dictionary.datamodule)

        return cls(keypointnet=keypointnet, optimizer=optimizer, loss=loss, trainer=trainer, datamodule=datamodule)
