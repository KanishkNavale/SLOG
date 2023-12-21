import unittest
import torch

from slog import geometry
from slog.datatypes import Pose


class TestGeometry(unittest.TestCase):

    def test_compute_pose(self):

        points = torch.as_tensor([[1.0, 0.0, 0.0],
                                  [2.0, 0.0, 0.0],
                                  [3.0, 0.0, 0.0],
                                  [4.0, 0.0, 0.0]], dtype=torch.float32)

        pose = geometry.compute_pose(points)

        self.assertTrue(torch.allclose(pose.translation, torch.mean(points, dim=0)))
        self.assertTrue(torch.allclose(pose.quaternion, torch.as_tensor([1.0, 0.0, 0.0, 0.0])))
        self.assertTrue(torch.allclose(pose.rotation, torch.eye(3)))

        # Test SVD Solver
        svd_based_pose = geometry.compute_pose(points)
        self.assertTrue(torch.allclose(pose.translation, svd_based_pose.translation))

    def test_relative_pose(self):
        points = torch.as_tensor([[1.0, 0.0, 0.0],
                                  [2.0, 0.0, 0.0],
                                  [3.0, 0.0, 0.0],
                                  [4.0, 0.0, 0.0],
                                  [5.0, 0.0, 0.0]], dtype=torch.float32)

        # Init. Random Pose
        random_pose = Pose(translation=torch.as_tensor([0.45701192, 0.06827693, 0.93321257]),
                           quaternion=torch.as_tensor([1.0, 0.0, 0.0, 0.0]))

        # Apply transformation
        displaced_points = random_pose.translation + (random_pose.rotation @ points.T).T

        relative_pose = geometry.compute_relative_pose(displaced_points, points)

        self.assertTrue(torch.allclose(random_pose.rigid_body_transformation, relative_pose.rigid_body_transformation))


if __name__ == '__main__':
    unittest.main()
