from slog.keypoint_nets import KeypointNetworkTrainer

if __name__ == "__main__":
    trainer = KeypointNetworkTrainer(yaml_config_path="sandbox/keypointnet/keypointnet-config.yaml")
    trainer.fit()
