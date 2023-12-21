from __future__ import annotations
from typing import List, Dict, Any, Tuple, Union
from dataclasses import dataclass

import numpy as np
import torch
from kornia.geometry import conversions as transform
from kornia.geometry.conversions import QuaternionCoeffOrder

from slog import utils
from slog.inference import is_rotationmatrix


@dataclass
class Keypoint:
    __slots__ = ['descriptor', 'location']

    descriptor: torch.Tensor
    location: torch.Tensor

    def to_dict(self) -> Dict[str, List[Any]]:
        return {"descriptor": self.descriptor_as_numpy.tolist(), "vu": self.location_as_numpy}

    @property
    def descriptor_as_numpy(self) -> np.ndarray:
        return utils.convert_tensor_to_numpy(self.descriptor)

    @property
    def descriptor_as_string(self) -> np.ndarray:
        return f'Descriptor: {np.around(self.descriptor_as_numpy.tolist(), 4)}'

    @property
    def location_as_numpy(self) -> Tuple[int, int]:
        return utils.convert_tensor_to_numpy(self.location).ravel().astype(np.int64).tolist()

    @classmethod
    def from_dictionary(cls, dictionary: Dict[Any, Any], device: torch.device = torch.device('cpu')) -> Keypoint:
        descriptor = torch.as_tensor(np.array(dictionary["descriptor"]), dtype=torch.float32, device=device)
        location = torch.as_tensor(np.array(dictionary["vu"]).reshape(2, 1), dtype=torch.float32, device=device)

        return cls(descriptor, location)

    def __post_init__(self):
        if self.location.shape != (2, 1):
            raise ValueError("Location of Keypoint in 'vu' format is not of shape (2,)")


@dataclass
class KeypointPrediction:
    __slots__ = ['keypoint', 'confidence']

    keypoint: Keypoint
    confidence: torch.Tensor

    def to_dict(self) -> Dict[str, Any]:
        keypoint_dictionary = self.keypoint.to_dict()
        confidence_dictionary = {'confidence': self.confidence.item()}
        dictionary = utils.merge_dictionaries([keypoint_dictionary, confidence_dictionary])

        return {'KeypointPrediction': dictionary}


@dataclass
class Pose:
    __slots__ = ['translation', 'quaternion']

    translation: torch.Tensor
    quaternion: torch.Tensor

    @classmethod
    def from_translation_and_rotation(cls, translation: torch.Tensor, rotation: torch.Tensor) -> Pose:

        if not is_rotationmatrix(rotation):
            raise ValueError("The rotation matrix doesn't suffice the rotational matrix properties")

        quaternion = transform.rotation_matrix_to_quaternion(rotation.contiguous(), order=QuaternionCoeffOrder.WXYZ)

        return cls(translation, quaternion)

    @classmethod
    def from_rigid_body_transformation(cls, rigid_body_transformation: torch.Tensor) -> Pose:
        translation = rigid_body_transformation[:3, 3]
        rotation = rigid_body_transformation[:3, :3]

        if not is_rotationmatrix(rotation):
            raise ValueError("The rotation matrix doesn't suffice the rotational matrix properties")

        quaternion = transform.rotation_matrix_to_quaternion(rotation.contiguous(), order=QuaternionCoeffOrder.WXYZ)

        return cls(translation, quaternion)

    @property
    def rotation(self) -> torch.Tensor:
        rotation = transform.quaternion_to_rotation_matrix(self.quaternion, order=QuaternionCoeffOrder.WXYZ)
        return rotation.squeeze(dim=0)

    @property
    def rigid_body_transformation(self) -> torch.Tensor:

        rotation_translation = torch.hstack((self.rotation, self.translation[:, None]))
        padding = torch.as_tensor([0.0, 0.0, 0.0, 1.0],
                                  dtype=rotation_translation.dtype,
                                  device=rotation_translation.device)

        return torch.vstack((rotation_translation, padding))

    @property
    def as_string(self) -> str:
        return f'POSE - T:{self.translation_as_numpy} Q:{self.quaternion_as_numpy}'

    @property
    def translation_as_numpy(self) -> np.ndarray:
        return utils.convert_tensor_to_numpy(self.translation)

    @property
    def quaternion_as_numpy(self) -> np.ndarray:
        return utils.convert_tensor_to_numpy(self.quaternion)

    def to_dict(self) -> Dict[str, Any]:
        return {'Pose': {'Translation': self.translation_as_numpy.tolist(), 'Quaternion': self.quaternion_as_numpy.tolist()}}

    @classmethod
    def from_dictionary(cls, dictionary: Dict[Any, Any], device: torch.device = torch.device('cpu')) -> Pose:
        translation = torch.as_tensor(np.array(dictionary["Pose"]["Translation"]), dtype=torch.float32, device=device)
        quaternion = torch.as_tensor(np.array(dictionary["Pose"]["Quaternion"]), dtype=torch.float32, device=device)

        return cls(translation, quaternion)


@dataclass
class KeypointGraph:
    __slots__ = ['keypoint_list']

    keypoint_list: List[Keypoint]

    @property
    def pairwise_distances(self) -> torch.Tensor:

        graph: List[torch.Tensor] = []
        for a in self.keypoint_list:
            graph.append(torch.vstack([torch.dist(a.location, b.location) for b in self.keypoint_list]))

        return torch.hstack(graph)

    @property
    def pairwise_distances_as_numpy(self) -> np.ndarray:
        return utils.convert_tensor_to_numpy(self.pairwise_distances)

    def to_dict(self) -> Dict[str, Any]:
        keypoints_dict = [keypoint.to_dict() for keypoint in self.keypoint_list]

        if len(keypoints_dict) > 1:
            return {
                'KeypointGraph': {
                    'Keypoints': keypoints_dict,
                    'Graph': self.pairwise_distances_as_numpy.tolist()}
            }
        else:
            return {'KeypointGraph': {'Keypoints': keypoints_dict}}

    @classmethod
    def from_dictionary(cls, dictionary: Dict[Any, Any], device: torch.device = torch.device('cpu')) -> KeypointGraph:
        keypoint_list = [Keypoint.from_dictionary(keypoint, device)
                         for keypoint in dictionary["KeypointGraph"]["Keypoints"]]

        return cls(keypoint_list)


@dataclass
class KeypointPoseGraph:
    __slots__ = ['keypoint_graph', 'pose']

    keypoint_graph: KeypointGraph
    pose: Pose

    @property
    def fetch_keypoints(self) -> List[Keypoint]:
        return [keypoint for keypoint in self.keypoint_graph.keypoint_list]

    def to_dict(self) -> Dict[Any, Any]:
        keypoint_graph_dictionary = self.keypoint_graph.to_dict()

        if self.pose is not None:
            pose_dictionary = self.pose.to_dict()
        else:
            pose_dictionary = {}

        dictionary = utils.merge_dictionaries([keypoint_graph_dictionary, pose_dictionary])

        return {'KeypointPoseGraph': dictionary}

    @classmethod
    def from_dictionary(cls, dictionary: Dict[Any, Any], device: torch.device = torch.device('cpu')) -> KeypointPoseGraph:

        keypoint_graph = KeypointGraph.from_dictionary(dictionary["KeypointPoseGraph"], device)
        pose = Pose.from_dictionary(dictionary["KeypointPoseGraph"], device)

        return cls(keypoint_graph, pose)


@dataclass
class VisionInputData:
    _slots__ = ['rgb', 'depth', 'intrinsic', 'extrinsic', 'descriptor', 'mask']

    rgb: torch.Tensor
    depth: torch.Tensor
    intrinsic: torch.Tensor
    extrinsic: Pose
    descriptor: Union[torch.Tensor, None] = None
    mask: Union[torch.Tensor, None] = None

    @property
    def rgbd(self) -> torch.Tensor:
        return torch.concat([self.rgb, self.depth.unsqueeze(dim=-1)], dim=-1)

    def __post_init__(self):
        if not self.rgb.shape[-1] == 3:
            raise ValueError("The RGB initialized image does not have 3-Channels")

        if not self.intrinsic.shape == (3, 3):
            raise ValueError("Intrinsic Matrix size is not (3 x 3)")

        if torch.allclose(torch.linalg.det(self.intrinsic), torch.zeros(1, dtype=self.intrinsic.dtype, device=self.intrinsic.device)):
            raise ValueError("Intrinsic Matrix is not invertible")

        if self.descriptor is not None:
            if not self.rgb.shape[:2] == self.descriptor.shape[:2]:
                raise ValueError("RGB Image (H x W) is not equal Descriptor Image (H x W)")

        if self.mask is not None:
            if not self.rgb.shape[:2] == self.mask.shape[:2]:
                raise ValueError("RGB Image (H x W) is not equal Mask Image (H x W)")
