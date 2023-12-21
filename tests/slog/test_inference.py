import unittest
import torch

from slog import inference


class TestInference(unittest.TestCase):

    def test_is_rotation_matrix(self):

        rotation_matrix = torch.eye(3)
        random_matrix = torch.ones((3, 3))

        self.assertTrue(inference.is_rotationmatrix(rotation_matrix))
        self.assertFalse(inference.is_rotationmatrix(random_matrix))

    def test_keypoint_expectation(self):

        tensor = torch.zeros((3, 3, 3))
        tensor[0][0] = torch.ones(3)
        keypoint = torch.ones(3)
        temperature = torch.as_tensor(2.5)

        keypoint_expectation = inference.compute_keypoint_expectation(tensor, keypoint, temperature)
        max_expectation = torch.where(keypoint_expectation == keypoint_expectation.max())

        self.assertTrue(torch.allclose(torch.as_tensor([0, 0]), torch.hstack(max_expectation)))

    def test_confident_keypoint_expectation(self):
        tensor = torch.zeros((3, 3, 3))
        tensor[0][0] = torch.ones(3)
        tensor[1][1] = torch.ones(3) - 0.2

        keypoint = torch.ones(3)
        temperature = torch.as_tensor(2.5)
        confidence_cutoff = torch.as_tensor(0.9)

        keypoint_expectation = inference.compute_keypoint_confident_expectation(tensor, keypoint, temperature, confidence_cutoff)

        self.assertEqual(torch.tensor(1), keypoint_expectation[0][0])
        self.assertGreaterEqual(keypoint_expectation[1][1], confidence_cutoff)

        stripped_confident_distances = keypoint_expectation
        stripped_confident_distances[0][0] = torch.zeros(1)
        stripped_confident_distances[1][1] = torch.zeros(1)

        self.assertTrue(torch.allclose(torch.zeros((3, 3)), stripped_confident_distances))


if __name__ == '__main__':
    unittest.main()
