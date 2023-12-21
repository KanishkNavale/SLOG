import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, RichProgressBar

from slog.ende_zu_ende.keypointnet_informed_don import KeypointNetInformedDONConfig, KeypointNetInformedDON, DataModuleKeypointNetInformedDON

from slog.utils import initialize_config_file


class KeypointInformedDONTrainer:

    def __init__(self, yaml_config_path: str) -> None:

        config_dictionary = initialize_config_file(yaml_config_path)
        config = KeypointNetInformedDONConfig.from_dictionary(config_dictionary)

        # Init. checkpoints here,
        model_checkpoints = ModelCheckpoint(monitor='val_loss',
                                            filename=self._generate_model_name(config),
                                            dirpath=config.trainer.model_path,
                                            mode='min',
                                            save_top_k=1,
                                            verbose=True)
        bar = RichProgressBar()
        checkpoints = [model_checkpoints, bar]

        # Init. model and datamodule
        self.datamodule = DataModuleKeypointNetInformedDON(yaml_config_path)
        self.datamodule.prepare_data()
        self.datamodule.setup()
        self.model = KeypointNetInformedDON(yaml_config_path)

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

    @staticmethod
    def _generate_model_name(config: KeypointNetInformedDONConfig) -> str:

        don_details = f"don-{config.don.backbone}-d{config.don.descriptor_dimension}-{config.loss.don}"
        keypoint_details = f"keypointnet-{config.keypointnet.backbone}-d{config.keypointnet.bottleneck_dimension}-nc{config.keypointnet.n_keypoints}"

        return f"KID-{don_details}-{keypoint_details}"
