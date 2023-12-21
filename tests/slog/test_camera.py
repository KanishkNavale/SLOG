import unittest

import torch

from slog import camera


class TestCamera(unittest.TestCase):

    # Groundtruth Data
    uv = torch.as_tensor([[100], [200]], dtype=torch.float32)
    depth = torch.ones((1), dtype=torch.float32)
    intrinsic = torch.eye((3), dtype=torch.float32)
    extrinsic = torch.eye((4), dtype=torch.float32)
    camera_coordinates = torch.as_tensor([[100], [200], [1]], dtype=torch.float32)
    world_coordinates = torch.as_tensor([[100], [200], [1]], dtype=torch.float32)

    def test_pixel_to_camera_coordinates(self):
        cam_coords = camera.pixel_to_camera_coordinates(self.uv, self.depth, self.intrinsic)
        self.assertTrue(torch.allclose(cam_coords, self.camera_coordinates))

    def test_camera_to_world_coordinates(self):
        world_coords = camera.camera_to_world_coordinates(self.camera_coordinates, self.extrinsic)
        self.assertTrue(torch.allclose(world_coords, self.world_coordinates))

    def test_pixel_to_world_coordinates(self):
        world_coords = camera.pixel_to_world_coordinates(self.uv, self.depth, self.intrinsic, self.extrinsic)
        self.assertTrue(torch.allclose(world_coords, self.world_coordinates))

    def test_camera_to_pixel_coordinates(self):
        pixel_coords = camera.camera_to_pixel_coordinates(self.camera_coordinates, self.intrinsic)
        self.assertTrue(torch.allclose(pixel_coords, self.uv))

    def test_world_to_pixel_coordinates(self):
        world_coords = camera.world_to_pixel_coordinates(self.world_coordinates, self.intrinsic, self.extrinsic)
        self.assertTrue(torch.allclose(world_coords, self.uv))


if __name__ == '__main__':
    unittest.main()
