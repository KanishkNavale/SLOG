import unittest
import torch
import json

from slog import PoseGraphPredictor
from slog.datatypes import VisionInputData, KeypointPoseGraph


class TestPoseGraphPredictorApp(unittest.TestCase):

    def build_vision_input_data(self) -> VisionInputData:
        image = torch.zeros((200, 200, 4), dtype=torch.float32)

        image[10][11] = 1 * torch.as_tensor([10, 10, 10, 10], dtype=torch.float32)
        image[10][21] = 2 * torch.as_tensor([10, 10, 10, 10], dtype=torch.float32)
        image[20][11] = 3 * torch.as_tensor([10, 10, 10, 10], dtype=torch.float32)
        image[20][21] = 4 * torch.as_tensor([10, 10, 10, 10], dtype=torch.float32)

        image[110][111] = 2 * torch.as_tensor([10, 10, 10, 10], dtype=torch.float32)
        image[110][121] = 4 * torch.as_tensor([10, 10, 10, 10], dtype=torch.float32)
        image[120][111] = 1 * torch.as_tensor([10, 10, 10, 10], dtype=torch.float32)
        image[120][121] = 3 * torch.as_tensor([10, 10, 10, 10], dtype=torch.float32)

        image = torch.hstack((image, image))

        depth = torch.ones_like(image)[..., 0]
        intrinsics = torch.as_tensor([[600, 0.0, 100],
                                      [0.0, 600, 200],
                                      [0.0, 0.0, 1.0]], dtype=torch.float32)
        extrinsics = torch.eye((4), device=image.device)

        return VisionInputData(image[..., :3], depth, intrinsics, extrinsics, image)

    def build_occluded_vision_input_data(self) -> VisionInputData:
        image = torch.zeros((200, 200, 4), dtype=torch.float32)

        image[10][11] = 1 * torch.as_tensor([10, 10, 10, 10], dtype=torch.float32)
        # random occlusion #image[10][21] = 2 * torch.as_tensor([10, 10, 10, 10], dtype=torch.float32)
        image[20][11] = 3 * torch.as_tensor([10, 10, 10, 10], dtype=torch.float32)
        image[20][21] = 4 * torch.as_tensor([10, 10, 10, 10], dtype=torch.float32)

        # random occlusion #image[110][111] = 2 * torch.as_tensor([10, 10, 10, 10], dtype=torch.float32)
        image[110][121] = 4 * torch.as_tensor([10, 10, 10, 10], dtype=torch.float32)
        image[120][111] = 1 * torch.as_tensor([10, 10, 10, 10], dtype=torch.float32)
        image[120][121] = 3 * torch.as_tensor([10, 10, 10, 10], dtype=torch.float32)

        image = torch.hstack((image, image))

        depth = torch.ones_like(image)[..., 0]
        intrinsics = torch.as_tensor([[600, 0.0, 100],
                                      [0.0, 600, 200],
                                      [0.0, 0.0, 1.0]], dtype=torch.float32)
        extrinsics = torch.eye((4), device=image.device)

        return VisionInputData(image[..., :3], depth, intrinsics, extrinsics, image)

    def test_prediction_consistency(self):

        vision_data = self.build_vision_input_data()
        data = json.load(open('tests/data/keypoint_pose_graphs.json'))
        keypoint_pose_graph = KeypointPoseGraph.from_dictionary(data[0], device=vision_data.rgb.device)

        pose_predictor = PoseGraphPredictor(-1.0, 0.5, 0.1, debug_path='/tmp')
        prediction_list = pose_predictor.run(vision_data, keypoint_pose_graph)

        self.assertEqual(len(prediction_list), 4)

    def test_occluded_prediction_consistency(self):

        vision_data = self.build_occluded_vision_input_data()
        data = json.load(open('tests/data/keypoint_pose_graphs.json'))
        keypoint_pose_graph = KeypointPoseGraph.from_dictionary(data[0], device=vision_data.rgb.device)

        pose_predictor = PoseGraphPredictor(-1.0, 0.5, 0.1, debug_path='/tmp')
        prediction_list = pose_predictor.run(vision_data, keypoint_pose_graph)

        self.assertEqual(len(prediction_list), 4)


if __name__ == '__main__':
    unittest.main()
