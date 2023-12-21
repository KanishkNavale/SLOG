from __future__ import annotations
from typing import Dict, Any, Union

from dataclasses import dataclass
import os


@dataclass
class DON:
    descriptor_dimension: int
    backbone: str

    @classmethod
    def from_dictionary(cls, dictionary: Dict[str, Any]) -> DON:
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
    name: str
    reduction: str
    temperature: Union[float, None] = None

    @classmethod
    def from_dictionary(cls, dictionary: Dict[str, Any]) -> Loss:
        return cls(**dictionary)

    def __post_init__(self):
        if self.name not in ["pixelwise_correspondence_loss",
                             "pixelwise_ntxent_loss"]:
            raise NotImplementedError(f"The specified loss function: {self.name} is not implemented")

        if self.name == "pixelwise_ntxent_loss" and self.temperature is None:
            raise ValueError(f"The loss function: {self.name} needs a temperatute scalar to be initialized")

        if self.reduction not in ["sum", "mean"]:
            raise NotImplementedError(f"The specified loss function: {self.reduction} is not implemented")


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
    mask_directory: str

    test_size: float
    shuffle: int

    n_workers: int
    batch_size: int
    n_correspondence: int

    @classmethod
    def from_dictionary(cls, dictionary: Dict[str, Any]) -> Trainer:
        return cls(**dictionary)

    def __post_init__(self):
        self.rgb_directory = os.path.abspath(self.rgb_directory)
        self.mask_directory = os.path.abspath(self.mask_directory)


@dataclass
class DenseObjectNetConfig:

    don: DON
    optimizer: Optimizer
    loss: Loss
    trainer: Trainer
    datamodule: DataModule

    @classmethod
    def from_dictionary(cls, dictionary: Dict[str, Any]) -> DenseObjectNetConfig:
        don = DON.from_dictionary(dictionary.don)
        optimizer = Optimizer.from_dictionary(dictionary.optimizer)
        loss = Loss.from_dictionary(dictionary.loss)
        trainer = Trainer.from_dictionary(dictionary.trainer)
        datamodule = DataModule.from_dictionary(dictionary.datamodule)

        return cls(don=don, optimizer=optimizer, loss=loss, trainer=trainer, datamodule=datamodule)
