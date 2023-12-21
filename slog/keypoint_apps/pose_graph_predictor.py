from typing import List, Union, Tuple
from sklearn.utils import shuffle
import itertools

import torch
import numpy as np
import cv2

from slog.inference import compute_keypoint_confident_expectation as compute_confident_probabilites
from slog.datatypes import KeypointGraph, KeypointPoseGraph, Keypoint, KeypointPrediction, VisionInputData, Pose
from slog import renderers, utils, camera


class PoseGraphPredictor:

    def __init__(self,
                 temperature: float,
                 confidence: float,
                 pruning_margin: float,
                 debug_path: str) -> None:

        confidence = np.clip(confidence, 1.0, 0.1)

        self.temperature = torch.as_tensor(temperature, dtype=torch.float32)
        self.confidence = torch.as_tensor(confidence, dtype=torch.float32)
        self.pruning_margin = pruning_margin

        # CPU copies
        self.cpu_temperature = temperature

        self.debug_path: str = debug_path

    def _dump_computed_expectations(self,
                                    file_index: int,
                                    spatial_expectation: torch.Tensor,
                                    keypoint: Keypoint) -> None:
        """Dumps a keypoint's spatial expectation to image.
        Args:
            file_index (int): filename prefix
            spatial_expectation (torch.Tensor): Spatial expectation of keypoint
            keypoint (Keypoint): Keypoint data
        """
        font = cv2.FONT_HERSHEY_SIMPLEX
        indices = torch.where(spatial_expectation >= self.confidence)
        offset = spatial_expectation.shape[0]

        image = spatial_expectation.unsqueeze(dim=-1).tile(1, 1, 3)
        image = utils.convert_tensor_to_cv2(image)

        for v, u in zip(indices[0], indices[1]):
            u = int(u.item())
            v = int(v.item())
            confidence = utils.convert_tensor_to_numpy(spatial_expectation[v][u])

            cv2.circle(image, (u, v), radius=1, color=(255, 0, 0), thickness=2)
            image = cv2.putText(image, f'{v, u}', (u + 5, v - 5), font, 0.3, (255, 0, 0), 1, cv2.LINE_AA)
            image = cv2.putText(image, f'Conf.: {confidence}', (u + 5, v + 5), font, 0.3, (0, 0, 255), 1, cv2.LINE_AA)

        image = cv2.putText(image, f'{keypoint.descriptor_as_string}',
                            (5, offset - 25), font, 0.3, (0, 255, 0), 1, cv2.LINE_AA)
        image = cv2.putText(image, f'Temp.: {self.cpu_temperature}',
                            (5, offset - 15), font, 0.3, (0, 255, 0), 1, cv2.LINE_AA)

        utils.dump_as_image(self.debug_path, f'Keypoint_{file_index}.png', image)

    def _compute_keypoints_expectation(self,
                                       descriptors: torch.Tensor,
                                       keypoint_list: List[Keypoint]) -> List[List[KeypointPrediction]]:
        """Compute keypoint's spatial expectation from a list keypoints.
        Args:
            descriptors (torch.Tensor): Descriptor image
            keypoint_list (List[Keypoint]): list of 'Keypoint' datatype
        Returns:
            List[List[KeypointPrediction]]: list of clusters (list of KeypointPrediction datatype)
        """
        # Keypoints Prediction List
        keypoints_prediction_list: List[List[KeypointPrediction]] = []

        if self.temperature.device is not descriptors.device:
            self.temperature.to(descriptors.device)

        if self.confidence.device is not descriptors.device:
            self.confidence.to(descriptors.device)

        for i, keypoint in enumerate(keypoint_list):

            # Moving tensors to same device as the descriptor image
            keypoint.descriptor.to(descriptors.device)

            spatial_expectation = compute_confident_probabilites(descriptors,
                                                                 keypoint.descriptor,
                                                                 self.temperature,
                                                                 self.confidence)

            # dump expectations
            self._dump_computed_expectations(i, spatial_expectation, keypoint)

            us, vs = torch.where(spatial_expectation >= self.confidence)
            keypoint_predictions = [KeypointPrediction(Keypoint(keypoint.descriptor, torch.vstack((v, u)).type(torch.float32)),
                                                       spatial_expectation[u][v]) for u, v in zip(us, vs)]
            keypoints_prediction_list.append(keypoint_predictions)

        return keypoints_prediction_list

    def _compute_pairwise_distances(self,
                                    keypoint_list: List[KeypointPrediction]) -> torch.Tensor:
        """Computes pairwise distances of KeypointPrediction datatype in a list.
        Returns:
            torch.Tensor: pairwise distances
        """
        graph: List[torch.Tensor] = []
        for a in keypoint_list:
            graph.append(torch.vstack([torch.dist(a.keypoint.location, b.keypoint.location) for b in keypoint_list]))
        return torch.hstack(graph)

    def _prune_clusters(self,
                        actual_pairwise_distances: torch.Tensor,
                        keypoint_list: List[KeypointPrediction]) -> List[KeypointPrediction]:
        """Delete clusters formed with a keypoint or more not belonging to the KeypointPoseGraph definition.
        Args:
            actual_pairwise_distances (torch.Tensor): pairwise distance from reference KeypoitPoseGraph datatype.
            keypoint_list (List[KeypointPrediction]): list of KeypointPrediction -> cluster
        Returns:
            List[KeypointPrediction]: valid cluster
        """
        predicted_pairwise_distances = self._compute_pairwise_distances(keypoint_list)

        # Compute number of mismatches
        mismatch_distance = actual_pairwise_distances.flatten() - predicted_pairwise_distances.flatten()
        mismatch_distance = utils.convert_tensor_to_numpy(mismatch_distance[:len(keypoint_list)])

        # Compile a list of matched keypoints
        matched_keypoint_list: List[KeypointPrediction] = []
        for i, keypoint in enumerate(keypoint_list):
            if mismatch_distance[i] == 0.0:
                matched_keypoint_list.append(keypoint)

        return matched_keypoint_list

    def _build_combination_of_clusters(self,
                                       pose_graph: KeypointPoseGraph,
                                       keypoints_prediction_list: List[List[KeypointPrediction]]) -> List[List[KeypointPrediction]]:
        """Builds all possible cluster keypoints from list of list of Keypoints
        !!! GOD SAVE THIS PC :) !!!
        Args:
            pose_graph (KeypointPoseGraph): KeypointPoseGraph datatype as tempate to build the graphs
            keypoints_prediction_list (List[List[KeypointPrediction]]): list of list of KeypointPrediction datatype
        Returns:
            List[List[KeypointPrediction]]: List of clusters (List of KeypointPrediction)
        """
        print("PoseGraphPredictor: Building all cluster possibilies")

        pairwise_distances = pose_graph.keypoint_graph.pairwise_distances

        cluster_possibilities = list(itertools.product(*keypoints_prediction_list))

        approximated_clusters: List[KeypointPrediction] = []
        for cluster in cluster_possibilities:
            pruned_cluster = self._prune_clusters(pairwise_distances, cluster)
            if len(pruned_cluster) >= 3:  # if no. of keypoints < 3 then pose cannot be computed
                approximated_clusters.append(pruned_cluster)

        return approximated_clusters

    def _compute_cam_coords(self, cluster: List[KeypointPrediction], input_data: VisionInputData) -> torch.Tensor:
        """Computes cam coordinates of keypoint's uv position in cluster
        Args:
            cluster (List[KeypointPrediction]): cluster
        Returns:
            torch.Tensor: Camera Coordinates of dimension (N x 3)
        """
        cam_coords: List[torch.Tensor] = []
        for keypoint_prediction in cluster:
            v, u = keypoint_prediction.keypoint.location_as_numpy
            depth = input_data.depth[u][v]
            cam_coords.append(camera.pixel_to_camera_coordinates(
                keypoint_prediction.keypoint.location, depth, input_data.intrinsic))

        return torch.hstack(cam_coords).T

    def _validate_clusters(self,
                           clusters: List[List[KeypointPrediction]],
                           input_data: VisionInputData) -> List[List[KeypointPrediction]]:
        """Validates a list of clusters by merging clusters having same and partial information belonging to same cluster.
        Args:
            clusters (List[List[KeypointPrediction]]): list of clusters
            input_data (VisionInputData): Vision Input Data
        Returns:
            List[List[KeypointPrediction]]: list of valid clusters
        """
        print("PoseGraphPredictor: Validation all clusters")

        # Shuffle clusters -> robust!
        clusters = shuffle(clusters)

        # Compute cluster bipairs
        similar_cluster_bipairs: List[Union[int, set]] = []
        for i, cluster_a in enumerate(clusters):
            for j, cluster_b in enumerate(clusters):
                if j <= i:
                    continue

                cluster_a_center = torch.mean(self._compute_cam_coords(cluster_a, input_data), axis=0)
                cluster_b_center = torch.mean(self._compute_cam_coords(cluster_b, input_data), axis=0)

                if torch.dist(cluster_a_center, cluster_b_center) < self.pruning_margin:
                    similar_cluster_bipairs.append([i, j])

        # Combine similar bipairs
        similar_cluster_bipairs = [set(bipair) for bipair in similar_cluster_bipairs]
        for a, b in itertools.product(similar_cluster_bipairs, similar_cluster_bipairs):
            if a.intersection(b):
                a.update(b)
                b.update(a)
        similar_cluster_bipairs = sorted([sorted(list(x)) for x in similar_cluster_bipairs])
        similar_clusters = list(pair for pair, _ in itertools.groupby(similar_cluster_bipairs))

        # Compute cluster with more keypoints in similar cluster group
        unique_cluster_index: List[int] = []
        for cluster in similar_clusters:
            largest_intra_cluster_len: int = 0
            largest_intra_cluster_index: int = 0
            for _, cluster_index in enumerate(cluster):
                if len(clusters[cluster_index]) > largest_intra_cluster_len:
                    largest_intra_cluster_index = cluster_index
                    largest_intra_cluster_len = len(clusters[cluster_index])
            unique_cluster_index.append(largest_intra_cluster_index)

        unique_clusters = [clusters[index] for index in unique_cluster_index]

        return unique_clusters

    def _create_posegraph_datatype_from_clusters(self,
                                                 template_pose_graph: KeypointPoseGraph,
                                                 clusters: List[List[KeypointPrediction]],
                                                 input_data: VisionInputData) -> List[KeypointPoseGraph]:
        """Converts a list of clusters to list of KeypointPoseGraph datatype
           while assigning relative pose to the cluster w.r.t. template pose graph.
        Args:
            template_pose_graph (KeypointPoseGraph): KeypointPoseGraph as reference for pose.
            clusters (List[List[KeypointPrediction]]): list of clusters
            input_data (VisionInputData): vision input data
        Returns:
            List[KeypointPoseGraph]: list of KeypointPoseGraphs
        """
        # Fetch keypoints from the original cluster
        template_keypoints = template_pose_graph.fetch_keypoints
        template_keypoints = [KeypointPrediction(keypoint, 1.0)for keypoint in template_keypoints]

        # Fetch keypoints from the template cluster matching to the predicted cluster
        def match_template_kepoints_to_predicted_cluster(cluster: List[KeypointPrediction]) -> Tuple[List[KeypointPrediction], List[KeypointPrediction]]:
            matched_keypoints: List[KeypointPrediction] = []
            cluster_keypoints: List[KeypointPrediction] = []
            for cluster_keypoint_prediction in cluster:
                for template_keypoint_prediction in template_keypoints:
                    if torch.allclose(cluster_keypoint_prediction.keypoint.descriptor, template_keypoint_prediction.keypoint.descriptor):
                        matched_keypoints.append(template_keypoint_prediction)
                        cluster_keypoints.append(cluster_keypoint_prediction)
                        continue

            return cluster_keypoints, matched_keypoints

        print("PoseGraphPredictor: Converting clusters to KeypointPoseGraph datatype")

        pose_graphs: List[KeypointPoseGraph] = []
        for cluster in clusters:
            cluster_keypoints, template_keypoints = match_template_kepoints_to_predicted_cluster(cluster)

            template_cam_coords = self._compute_cam_coords(template_keypoints, input_data)
            cluster_cam_coords = self._compute_cam_coords(cluster_keypoints, input_data)

            cluster_center = torch.mean(cluster_cam_coords, axis=0)
            template_center = torch.mean(template_cam_coords, axis=0)

            zero_centered_cluster = cluster_cam_coords - cluster_center
            zero_centered_template = template_cam_coords - template_center

            covariance = (zero_centered_template.T @ zero_centered_cluster)

            U, _, VT = torch.linalg.svd(covariance)
            one_diag = torch.diag(torch.ones(VT.shape[0], device=U.device, dtype=U.dtype))
            one_diag[-1][-1] = torch.sign(torch.linalg.det(VT.T @ U))

            rotation = VT.T @ one_diag @ U

            pose_graphs.append(KeypointPoseGraph(KeypointGraph([keypoint_prediction.keypoint for keypoint_prediction in cluster]),
                               Pose.from_translation_and_rotation(cluster_center, rotation)))

        return pose_graphs

    def _dump_pose_graphs(self, list_pose_graphs: List[KeypointPoseGraph], input_data: VisionInputData) -> None:
        """Dumps the app processed information as .json & .png format
        Args:
            list_pose_graphs (List[KeypointPoseGraph]): list of KeypointPoseGraphs
            input_data (VisionInputData): vision input data
        """
        print("PoseGraphPredictor: Saving Data & Exiting")

        image = input_data.rgb[:, :, :3]  # Sanity check
        image = utils.convert_tensor_to_cv2(image)

        pose_graph_dictionary = []

        for pose_graph in list_pose_graphs:
            for keypoint in pose_graph.keypoint_graph.keypoint_list:
                v, u = keypoint.location_as_numpy
                cv2.circle(image, (v, u), radius=1, color=(255, 0, 0), thickness=2)

            image = renderers.render_pose(image, pose_graph.pose, input_data.intrinsic)
            pose_graph_dictionary.append(pose_graph.to_dict())

        utils.dump_as_image(self.debug_path, 'Predictions_PoseGraph.png', image)
        utils.dump_as_json(self.debug_path, 'Predictions_PoseGraphs.json', pose_graph_dictionary)

    def run(self,
            input_data: VisionInputData,
            pose_graph: KeypointPoseGraph) -> List[KeypointPoseGraph]:

        keypoints_expectation_list = self._compute_keypoints_expectation(input_data.descriptor,
                                                                         pose_graph.keypoint_graph.keypoint_list)

        cluster_possibilities = self._build_combination_of_clusters(pose_graph, keypoints_expectation_list)
        valid_clusters = self._validate_clusters(cluster_possibilities, input_data)
        pose_graphs = self._create_posegraph_datatype_from_clusters(pose_graph, valid_clusters, input_data)
        self._dump_pose_graphs(pose_graphs, input_data)

        return pose_graphs
