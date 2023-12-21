from slog.don import DenseObjectNetworkTrainer

if __name__ == "__main__":
    trainer = DenseObjectNetworkTrainer(yaml_config_path="sandbox/don/don-config.yaml")
    trainer.fit()