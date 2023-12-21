import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, RichProgressBar

from slog.keypoint_nets import KeypointNetConfig, KeypointNetwork, DataModuleKeypointNet
from slog.utils import initialize_config_file


class KeypointNetworkTrainer:

    def __init__(self, yaml_config_path: str) -> None:

        config_dictionary = initialize_config_file(yaml_config_path)
        config = KeypointNetConfig.from_dictionary(config_dictionary)

        backbone_name = config.keypointnet.backbone.replace("_", "")

        # Init. checkpoints here,
        model_checkpoints = ModelCheckpoint(monitor='Validation Loss',
                                            filename=f"keypointnet-{backbone_name}-d{config.keypointnet.backbone}-nc{config.keypointnet.n_keypoints}",
                                            dirpath=config.trainer.model_path,
                                            mode='min',
                                            save_top_k=1,
                                            verbose=True)
        bar = RichProgressBar()
        checkpoints = [model_checkpoints, bar]

        # Init. model and datamodule
        self.datamodule = DataModuleKeypointNet(yaml_config_path)
        self.datamodule.prepare_data()
        self.datamodule.setup()
        self.model = KeypointNetwork(yaml_config_path)

        self.trainer = pl.Trainer(logger=config.trainer.enable_logging,
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
