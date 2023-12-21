import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, RichProgressBar
from pytorch_lightning.loggers import TensorBoardLogger

from slog.don import DenseObjectNetConfig
from slog.don import DataModuleDON
from slog.don import DON

from slog.utils import initialize_config_file


class DenseObjectNetworkTrainer:

    def __init__(self, yaml_config_path: str) -> None:

        config_dictionary = initialize_config_file(yaml_config_path)
        config = DenseObjectNetConfig.from_dictionary(config_dictionary)

        # Init. checkpoints here,
        model_checkpoints = ModelCheckpoint(monitor='val_loss',
                                            filename=self._generate_model_name(config),
                                            dirpath=config.trainer.model_path,
                                            mode='min',
                                            save_top_k=1,
                                            verbose=True)
        bar = RichProgressBar()
        checkpoints = [model_checkpoints, bar]

        # Init. Logger
        if config.trainer.enable_logging:
            tensorboard_logger = TensorBoardLogger(config.trainer.tensorboard_path)
        else:
            tensorboard_logger = None

        # Init. model and datamodule
        self.datamodule = DataModuleDON(yaml_config_path)
        self.datamodule.prepare_data()
        self.datamodule.setup()
        self.model = DON(yaml_config_path)

        self.trainer = pl.Trainer(logger=tensorboard_logger,
                                  enable_checkpointing=config.trainer.enable_checkpointing,
                                  check_val_every_n_epoch=config.trainer.validation_frequency,
                                  max_epochs=config.trainer.epochs,
                                  log_every_n_steps=config.trainer.logging_frequency,
                                  default_root_dir=config.trainer.training_directory,
                                  callbacks=checkpoints,
                                  enable_model_summary=True,
                                  detect_anomaly=True,
                                  accelerator='auto',
                                  devices='auto')

    def fit(self) -> None:
        self.trainer.fit(self.model, datamodule=self.datamodule)

    @staticmethod
    def _generate_model_name(config: DenseObjectNetConfig) -> str:

        if config.loss.name == "pixelwise_correspondence_loss":
            loss = "correspondence"
        elif config.loss.name == "pixelwise_ntxent_loss":
            loss = "ntxent"

        return f"don-d{config.don.descriptor_dimension}-{config.don.backbone.replace('_', '')}-nc{config.datamodule.n_correspondence}-{loss}"
