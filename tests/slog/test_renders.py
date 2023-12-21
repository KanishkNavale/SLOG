import unittest

import torch
import numpy as np

from slog import renderers
from slog.datatypes import Pose


class TestRenders(unittest.TestCase):

    def test_render_spatial_distribution(self):
        image = torch.ones(256, 256, 3)
        heatmap = torch.ones(256, 256)

        heatmap_superposed_image = renderers.render_spatial_distribution(image, heatmap)

        self.assertEqual(heatmap_superposed_image.dtype, np.uint8)

    def test_render_pose(self):
        image = np.ones((256, 256, 3), dtype=np.uint8)
        pose = Pose(translation=torch.as_tensor([0.45701192, 0.06827693, 0.93321257]),
                    quaternion=torch.as_tensor([1.0, 0.0, 0.0, 0.0]))
        intrinsic = torch.as_tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=torch.float32)

        pose_superposed_image = renderers.render_pose(image, pose, intrinsic)
        self.assertEqual(pose_superposed_image.dtype, np.uint8)


if __name__ == '__main__':
    unittest.main()
