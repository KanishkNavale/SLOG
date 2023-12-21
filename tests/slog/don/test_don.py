import unittest
import torch
from slog.don import DON


class TestDON(unittest.TestCase):
    yaml_config_file = 'tests/data/don-config.yaml'
    dense_object_nets = DON(yaml_config_file)

    def test_init_cpu(self):
        self.assertTrue(isinstance(self.dense_object_nets, DON))
        self.assertTrue(self.dense_object_nets.device, torch.device('cpu'))

    def test_compute_dense_features(self):
        dummy_image = torch.ones((1, 3, 256, 256), dtype=torch.float32)
        prediction = self.dense_object_nets.compute_dense_features(dummy_image)

        self.assertTrue(prediction.shape[1] == self.dense_object_nets.config.don.descriptor_dimension)
        self.assertTrue(dummy_image.shape[2:] == prediction.shape[2:])


if __name__ == '__main__':
    unittest.main()
