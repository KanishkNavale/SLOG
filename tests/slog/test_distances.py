import unittest
import torch

from slog import distances


class TestDistances(unittest.TestCase):

    def test_l2(self):
        x = torch.as_tensor([0.0, 0.0])
        y = torch.as_tensor([1.0, 0.0])
        self.assertEqual(distances.l2(x, y), torch.as_tensor([1.0]))

    def test_l1(self):
        x = torch.as_tensor([10.0, 10.0])
        y = torch.as_tensor([1.0, 1.0])
        self.assertEqual(distances.l1(x, y), torch.as_tensor([18.0]))

    def test_guassian_distance_kernel(self):
        x = torch.as_tensor([0.0, 0.0])
        y = torch.as_tensor([1.0, 0.0])
        temperature = 1.0

        truth = 0.3679
        computed = distances.guassian_distance_kernel(x, y, temperature).item()
        self.assertAlmostEqual(truth, computed, places=4)

    def test_exp_guassian_distance_kernel(self):
        x = torch.as_tensor([0.0, 0.0])
        y = torch.as_tensor([1.0, 0.0])
        temperature = torch.as_tensor(1.0)

        truth = 0.6922
        computed = distances.exp_guassian_distance_kernel(x, y, temperature).item()
        self.assertAlmostEqual(truth, computed, places=4)


if __name__ == '__main__':
    unittest.main()
