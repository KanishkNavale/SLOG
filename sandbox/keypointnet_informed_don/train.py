from slog.ende_zu_ende.keypointnet_informed_don import KeypointInformedDONTrainer


if __name__ == "__main__":
    trainer = KeypointInformedDONTrainer("sandbox/keypointnet_informed_don/keypointnet-config.yaml")
    trainer.fit()
