import unittest
import torch
import numpy as np

from slog import datatypes


class TestDatatypes(unittest.TestCase):

    maxDiff = None

    def test_keypoint(self):
        descriptor = torch.ones((3,), dtype=torch.float32)
        vu = torch.as_tensor([[255], [255]], dtype=torch.float32)

        keypoint = datatypes.Keypoint(descriptor=descriptor, location=vu)

        # check properties
        self.assertTrue(isinstance(keypoint.descriptor_as_numpy, np.ndarray))
        self.assertEqual(keypoint.descriptor_as_string, 'Descriptor: [1. 1. 1.]')
        self.assertTrue(isinstance(keypoint.location_as_numpy, list))

        # check dictonary generation
        keypoint_dictionary = {"descriptor": [1.0, 1.0, 1.0], "vu": [255, 255]}
        self.assertDictEqual(keypoint.to_dict(), keypoint_dictionary)

        # init. keypoint from dictionary
        keypoint_init_from_dictionary = datatypes.Keypoint.from_dictionary(keypoint_dictionary, device=torch.device('cpu'))
        self.assertTrue(torch.allclose(keypoint.descriptor, keypoint_init_from_dictionary.descriptor))
        self.assertTrue(torch.allclose(keypoint.location, keypoint_init_from_dictionary.location))

    def test_keypointprediction(self):
        descriptor = torch.ones((3,), dtype=torch.float32)
        vu = torch.as_tensor([[255], [255]], dtype=torch.float32)
        keypoint = datatypes.Keypoint(descriptor=descriptor, location=vu)

        confidence = torch.randn(1).clamp(0.0, 1.0)

        keypoint_prediction = datatypes.KeypointPrediction(keypoint=keypoint, confidence=confidence)

        self.assertTrue(torch.allclose(keypoint_prediction.keypoint.descriptor, descriptor))
        self.assertTrue(torch.allclose(keypoint_prediction.keypoint.location, vu))
        self.assertTrue(torch.allclose(keypoint_prediction.confidence, confidence))

        # check dictionary prediction
        keypoint_prediction_dictionary = {'KeypointPrediction':
                                          {'descriptor': [1.0, 1.0, 1.0],
                                           'vu': [255, 255],
                                           'confidence': confidence.numpy()}
                                          }
        self.assertDictEqual(keypoint_prediction.to_dict(), keypoint_prediction_dictionary)

    def test_pose(self):
        translation = torch.zeros(3,)
        quaternion = torch.as_tensor([1.0, 0.0, 0.0, 0.0])

        pose = datatypes.Pose(translation=translation, quaternion=quaternion)

        # check properties
        self.assertTrue(pose.rotation.shape, (3, 3))
        self.assertTrue(pose.rigid_body_transformation.shape, (4, 4))
        self.assertEqual(pose.as_string, 'POSE - T:[0. 0. 0.] Q:[1. 0. 0. 0.]')
        self.assertTrue(isinstance(pose.translation_as_numpy, np.ndarray))
        self.assertTrue(isinstance(pose.quaternion_as_numpy, np.ndarray))

        # check dictonary generation
        pose_dictionary = {'Pose': {'Translation': [0.0, 0.0, 0.0], 'Quaternion': [1.0, 0.0, 0.0, 0.0]}}
        self.assertDictEqual(pose.to_dict(), pose_dictionary)

        # init. keypoint from dictionary
        pose_init_from_dictionary = datatypes.Pose.from_dictionary(pose_dictionary, device=torch.device('cpu'))
        self.assertTrue(torch.allclose(pose.translation, pose_init_from_dictionary.translation))
        self.assertTrue(torch.allclose(pose.quaternion, pose_init_from_dictionary.quaternion))

    def test_keypointgraph(self):
        keypoint_a = datatypes.Keypoint(descriptor=torch.ones(3,), location=torch.as_tensor([[0], [1]], dtype=torch.float32))
        keypoint_b = datatypes.Keypoint(descriptor=torch.zeros(3,), location=torch.as_tensor([[0], [0]], dtype=torch.float32))

        keypoint_graph = datatypes.KeypointGraph([keypoint_a, keypoint_b])

        # check properties
        self.assertTrue(torch.allclose(keypoint_graph.pairwise_distances, torch.as_tensor([[0.0, 1.0], [1.0, 0.0]])))
        self.assertTrue(isinstance(keypoint_graph.pairwise_distances_as_numpy, np.ndarray))

        # check dictionary generation
        keypoint_graph_dictionary = {"KeypointGraph":
                                     {
                                         "Keypoints": [
                                             {"descriptor": [1.0, 1.0, 1.0], "vu": [0, 1]},
                                             {"descriptor": [0.0, 0.0, 0.0], "vu": [0, 0]}
                                         ],
                                         "Graph": [[0.0, 1.0], [1.0, 0.0]]
                                     }
                                     }
        self.assertDictEqual(keypoint_graph.to_dict(), keypoint_graph_dictionary)

        # check init. from dicionary
        keypoint_graph_from_dictionary = datatypes.KeypointGraph.from_dictionary(keypoint_graph_dictionary, device=torch.device('cpu'))
        self.assertDictEqual(keypoint_graph.to_dict(), keypoint_graph_from_dictionary.to_dict())

    def test_keypoint_pose_graph(self):
        keypoint_a = datatypes.Keypoint(descriptor=torch.ones(3,), location=torch.as_tensor([[0], [1]], dtype=torch.float32))
        keypoint_b = datatypes.Keypoint(descriptor=torch.zeros(3,), location=torch.as_tensor([[0], [0]], dtype=torch.float32))
        keypoint_graph = datatypes.KeypointGraph([keypoint_a, keypoint_b])

        translation = torch.zeros(3,)
        quaternion = torch.as_tensor([1.0, 0.0, 0.0, 0.0])
        pose = datatypes.Pose(translation=translation, quaternion=quaternion)

        keypoint_pose_graph = datatypes.KeypointPoseGraph(keypoint_graph=keypoint_graph, pose=pose)

        # check properties
        self.assertEqual(len(keypoint_pose_graph.fetch_keypoints), 2)

        # check dictionary generation
        keypoint_pose_graph_dictionary = {
            "KeypointPoseGraph": {
                "KeypointGraph":
                {
                    "Keypoints": [
                        {"descriptor": [1.0, 1.0, 1.0], "vu": [0, 1]},
                        {"descriptor": [0.0, 0.0, 0.0], "vu": [0, 0]}
                    ],
                    "Graph": [[0.0, 1.0], [1.0, 0.0]]
                },
                "Pose": {'Translation': [0.0, 0.0, 0.0], 'Quaternion': [1.0, 0.0, 0.0, 0.0]}
            }
        }

        self.assertDictEqual(keypoint_pose_graph.to_dict(), keypoint_pose_graph_dictionary)

        # check init. from dictionary
        keypoint_pose_graph_from_dictionary = datatypes.KeypointPoseGraph.from_dictionary(keypoint_pose_graph_dictionary, device=torch.device('cpu'))
        self.assertDictEqual(keypoint_pose_graph.to_dict(), keypoint_pose_graph_from_dictionary.to_dict())

    def test_visioninputdata(self):
        rgb = torch.ones((255, 255, 3))
        depth = torch.ones((255, 255))
        intrinsic = torch.eye(3)
        extrinsic = datatypes.Pose(torch.zeros(3,), torch.as_tensor([1.0, 0.0, 0.0, 0.0]))
        descriptor = None

        vision_input_data_without_descriptor = datatypes.VisionInputData(rgb=rgb,
                                                                         depth=depth,
                                                                         intrinsic=intrinsic,
                                                                         extrinsic=extrinsic,
                                                                         descriptor=descriptor)
        self.assertTrue(isinstance(vision_input_data_without_descriptor, datatypes.VisionInputData))

        descriptor = torch.rand((255, 255, 16))
        vision_input_data_with_descriptor = datatypes.VisionInputData(rgb=rgb,
                                                                      depth=depth,
                                                                      intrinsic=intrinsic,
                                                                      extrinsic=extrinsic,
                                                                      descriptor=descriptor)
        self.assertTrue(isinstance(vision_input_data_with_descriptor, datatypes.VisionInputData))

        # check properties
        self.assertTrue(torch.allclose(vision_input_data_with_descriptor.rgbd, torch.concat([rgb, depth.unsqueeze(dim=-1)], dim=-1)))


if __name__ == '__main__':
    unittest.main()
