from typing import List

import cv2
import numpy as np
import torch

import pygame

from slog import camera, geometry, renderers
from slog.utils import dump_as_json
from slog.datatypes import Keypoint, KeypointPoseGraph, KeypointGraph, VisionInputData
from slog import utils
from slog.inference import compute_keypoint_confident_expectation as compute_confident_probabilites

pygame.init()


class PoseGraphGenerator:
    def __init__(self, input: VisionInputData) -> None:
        self.input = input

        self.temp = torch.as_tensor(2.0, device=input.descriptor.device, dtype=input.descriptor.dtype)
        self.confidence = torch.as_tensor(0.1, device=input.descriptor.device, dtype=input.descriptor.dtype)
        self.debug_path = ''

        # Set up display
        pygame.display.set_caption('Keypoint Pose Regressor App.')
        width, height = input.descriptor.shape[1], input.descriptor.shape[0]
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.flip()

        # CPU copies of tensors
        self.cpu_temp = self.temp.item()
        self.cpu_conf = self.confidence.item()
        self.cpu_image = utils.convert_tensor_to_cv2(self.input.rgb)

        # app defaults
        self.enable_spatial_expectation = False
        self.spat_probs = None

        # keypoint records
        self.keypoint_pose_graph_list: List[KeypointPoseGraph] = []
        self.keypoint_list: List[Keypoint] = []

    def _save_and_exit(self, event: pygame.event) -> bool:
        """Saves keypoint record in json format and exits the app

        Args:
            event (pygame.event): pygame event
        """
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_s:
                print("PoseGraphGenerator: Saving Data & Exiting")

                dump = [graph.to_dict() for graph in self.keypoint_pose_graph_list]
                dump_as_json(self.debug_path, 'keypoint_pose_graphs.json', dump)

                return True

    def _add_keypoints_as_graph(self, event: pygame.event) -> None:
        """Stores a list of current selected keypoints in self.keypoints_records

        Args:
            event (pygame.event): pygame event
        """
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                if len(self.keypoint_list) != 0:
                    if len(self.keypoint_list) >= 3:
                        pose = self._compute_pose(self.keypoint_list)
                        self.keypoint_pose_graph_list.append(KeypointPoseGraph(KeypointGraph(self.keypoint_list), pose))
                        self.keypoint_list = []
                    else:
                        self.keypoint_pose_graph_list.append(KeypointPoseGraph(KeypointGraph(self.keypoint_list), None))
                        self.keypoint_list = []

    def _clear_keypoint_graphs(self, event: pygame.event) -> None:
        """Clears a list of current selected keypoints in self.keypoints_records

        Args:
            event (pygame.event): pygame event
        """
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_t:
                self.keypoint_pose_graph_list = []
                self.keypoint_list = []

    def _record_keypoints(self, event: pygame.event) -> None:
        """Saves the selected keypoint in a list of self.keypoints

        Args:
            event (pygame.event): pygame event
        """
        if event.type == pygame.MOUSEBUTTONDOWN:
            u, v = pygame.mouse.get_pos()

            descriptor = self.input.descriptor[v][u]
            location = torch.as_tensor(np.vstack((u, v)), dtype=descriptor.dtype, device=descriptor.device)

            self.keypoint_list.append(Keypoint(descriptor, location))

    def _clear_keypoints(self, event: pygame.event) -> None:
        """Clears current selection of keypoints

        Args:
            event (pygame.event): pygame event
        """
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_r:
                self.keypoint_list = []

    def _update_display(self, image: np.ndarray) -> None:
        """Updates the pygame window with information

        Args:
            image (np.ndarray): image to render in np.uint8 format
        """
        pygame_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pygame_image = self._annotate_texts(pygame_image)
        pygame_image = pygame.surfarray.make_surface(pygame_image.swapaxes(0, 1))

        self.screen.blit(pygame_image, (0, 0))
        pygame.display.update()

    def _compute_spatial_expectation(self) -> np.ndarray:
        """Computes spatial expecation of keypoint

        Returns:
            np.ndarray: expecation of keypoint in np.uint8 format
        """
        v, u = pygame.mouse.get_pos()
        keypoint = self.input.descriptor[u][v]

        weights = compute_confident_probabilites(self.input.descriptor, keypoint, self.temp, self.confidence)
        spat_probs = renderers.render_spatial_distribution(self.input.rgb, weights)

        return spat_probs

    def _toggle_spatial_expectation(self, event: pygame.event) -> None:
        """Toggles state to compute spatial expectation

        Args:
            event (pygame.event): pygame event
        """
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_d:
                self.enable_spatial_expectation = not self.enable_spatial_expectation

    def _compute_pose(self, keypoint_list: List[Keypoint]) -> None:
        cam_coords: List[torch.Tensor] = []
        for keypoint in keypoint_list:
            v, u = keypoint.location_as_numpy
            depth = self.input.depth[u][v]
            cam_coords.append(camera.pixel_to_camera_coordinates(keypoint.location, depth, self.input.intrinsic))

        exp_pose = geometry.compute_pose(torch.hstack(cam_coords).T)
        return exp_pose

    def _annotate_texts(self, image: np.ndarray) -> np.ndarray:
        """Adds temperature text on to an image

        Args:
            image (np.ndarray): image

        Returns:
            np.ndarray: image annotated with temperature text
        """
        font = cv2.FONT_HERSHEY_SIMPLEX

        for keypoint in self.keypoint_list:
            u, v = keypoint.location_as_numpy
            cv2.circle(image, (u, v), radius=1, color=(255, 0, 0), thickness=2)
            image = cv2.putText(image, f'{v, u}', (u + 5, v - 5), font, 0.3, (255, 0, 0), 1, cv2.LINE_AA)

        if len(self.keypoint_list) >= 3:
            pose = self._compute_pose(self.keypoint_list)
            image = renderers.render_pose(image, pose, self.input.intrinsic)

        for graph in self.keypoint_pose_graph_list:
            for keypoint in graph.keypoint_graph.keypoint_list:
                u, v = keypoint.location_as_numpy
                cv2.circle(image, (u, v), radius=1, color=(255, 0, 255), thickness=2)
                image = cv2.putText(image, f'{v, u}', (u + 5, v - 5), font, 0.3, (255, 0, 255), 1, cv2.LINE_AA)

                if graph.pose is not None:
                    image = renderers.render_pose(image, graph.pose, self.input.intrinsic)

        if self.enable_spatial_expectation:
            temp = 'Temperature:' + str(np.around(self.cpu_temp, 4))
            image = cv2.putText(image, temp, (0, 15), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

            confidence = 'Confidence:' + str(np.around(self.cpu_conf, 2))
            image = cv2.putText(image, confidence, (0, 30), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

        return image

    def _update_temperature(self, event: pygame.event) -> None:
        """Updates temperature parameter of the kernel function
        Args:
            event (pygame.event): pygame event
        """
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_KP8:
                self.temp += 0.1
                self.cpu_temp += 0.1
            elif event.key == pygame.K_KP2:
                self.temp -= 0.1
                self.cpu_temp -= 0.1

    def _update_confidence(self, event: pygame.event) -> None:
        """Updates confidence parameter of the kernel function
        Args:
            event (pygame.event): pygame event
        """
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_KP4:
                self.confidence += 0.1
                self.cpu_conf += 0.1
            elif event.key == pygame.K_KP6:
                self.confidence -= 0.1
                self.cpu_conf -= 0.1

        self.cpu_conf = np.clip(self.cpu_conf, 0.0, 1.0)
        self.confidence = torch.clamp(self.confidence, 0.0, 1.0)

    def run(self) -> List[KeypointPoseGraph]:
        running = True
        image = None

        while running:
            for event in pygame.event.get():
                self._toggle_spatial_expectation(event)

                self._update_confidence(event)
                self._update_temperature(event)

                self._record_keypoints(event)
                self._clear_keypoints(event)

                self._add_keypoints_as_graph(event)
                self._clear_keypoint_graphs(event)

                if self.enable_spatial_expectation:
                    image = self._compute_spatial_expectation()
                if image is None or not self.enable_spatial_expectation:
                    image = self.cpu_image

                self._update_display(image)

                if event.type == pygame.QUIT:
                    running = False

                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False

                if self._save_and_exit(event) is not None and True:
                    running = False

        pygame.quit()

        return self.keypoint_pose_graph_list
