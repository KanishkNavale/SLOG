from slog.ende_zu_ende.don_informed_keypointnet import DONInformedKeypointnetTrainer


if __name__ == "__main__":
    trainer = DONInformedKeypointnetTrainer("sandbox/don_informed_keypointnet/keypointnet-config.yaml")
    trainer.fit()
