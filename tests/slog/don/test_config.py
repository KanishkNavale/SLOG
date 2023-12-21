import unittest

from slog.don import DenseObjectNetConfig
from slog.don.configurations.config import DON, DataModule, Optimizer, Trainer
from slog.utils import initialize_config_file


class TestDONConfig(unittest.TestCase):

    yaml_config_file = 'tests/data/don-config.yaml'
    yaml_parsed_as_dictionary = initialize_config_file(yaml_config_file)
    config = DenseObjectNetConfig.from_dictionary(yaml_parsed_as_dictionary)

    def test_init(self):
        self.assertTrue(isinstance(self.config, DenseObjectNetConfig))

    def test_don_subclass(self):
        self.assertTrue(isinstance(self.config.don, DON))

    def test_optimizer_subclass(self):
        self.assertTrue(isinstance(self.config.optimizer, Optimizer))

    def test_trainer_subclass(self):
        self.assertTrue(isinstance(self.config.trainer, Trainer))

    def test_datamodule_subclass(self):
        self.assertTrue(isinstance(self.config.datamodule, DataModule))


if __name__ == '__main__':
    unittest.main()
