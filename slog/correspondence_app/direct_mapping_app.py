from __future__ import annotations
from typing import Tuple

import cv2
import numpy as np
import torch

import pygame

from slog import camera, datatypes
from slog import utils
pygame.init()


class DirectCorrespondenceMappingApp:
    def __init__(self, image_a: datatypes.VisionInputData, image_b: datatypes.VisionInputData) -> None:
        self.image_a = image_a
        self.image_b = image_b

        # check if images are of same length
        if not self.image_a.rgbd.shape == self.image_b.rgbd.shape:
            raise ValueError("Images are not of same sizes")

        self.image = utils.convert_tensor_to_cv2(torch.hstack((self.image_a.rgb, self.image_b.rgb)))

        # Set up display
        pygame.display.set_caption('Direct Corresponding Mapping App')
        width, height = self.image.shape[1], self.image.shape[0]
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.flip()

    def _trace_keypoints(self, event: pygame.event) -> Tuple[torch.Tensor, torch.Tensor]:
        """Saves the selected keypoint in a list of self.keypoints

        Args:
            event (pygame.event): pygame event
        """
        u, v = pygame.mouse.get_pos()

        u = min(u, self.image_a.rgb.shape[1] - 1)
        v = min(v, self.image_a.rgb.shape[0] - 1)

        depth = self.image_a.depth[v][u]
        uva = torch.as_tensor(np.vstack((u, v)), dtype=depth.dtype, device=depth.device)

        world = camera.pixel_to_world_coordinates(uva, depth, self.image_a.intrinsic, self.image_a.extrinsic.rigid_body_transformation)
        uvb = camera.world_to_pixel_coordinates(world, self.image_b.intrinsic, self.image_b.extrinsic.rigid_body_transformation)

        return uva, uvb

    def _update_display(self, uva: torch.Tensor, uvb: torch.Tensor) -> None:
        image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        uva = uva.numpy().ravel().astype(np.int32).tolist()
        uvb = uvb.numpy().ravel().astype(np.int32).tolist()

        ua, va = uva
        ub, vb = uvb

        font = cv2.FONT_HERSHEY_SIMPLEX

        cv2.circle(image, (ua, va), radius=1, color=(0, 0, 255), thickness=2)
        image = cv2.putText(image, f'{va, ua}', (ua + 5, va - 5), font, 0.3, (0, 0, 255), 1, cv2.LINE_AA)

        cv2.circle(image, (ub + self.image_a.rgb.shape[1], vb), radius=1, color=(255, 0, 0), thickness=2)
        image = cv2.putText(
            image, f'{vb, ub}', (ub + self.image_a.rgb.shape[1] + 5, vb - 5), font, 0.3, (255, 0, 0), 1, cv2.LINE_AA)

        pygame_image = pygame.surfarray.make_surface(image.swapaxes(0, 1))

        self.screen.blit(pygame_image, (0, 0))
        pygame.display.update()

    def run(self):
        running = True

        while running:
            for event in pygame.event.get():
                cam_coord_a, cam_coord_b = self._trace_keypoints(event)
                self._update_display(cam_coord_a, cam_coord_b)

                if event.type == pygame.QUIT:
                    running = False

                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False

        pygame.quit()
