import blenderproc as bproc

from typing import List
import numpy as np
import cv2

bproc.init()


def generate_erode_mask(mask: np.ndarray):
    kernel = np.ones((5, 5), np.uint8)
    erode = cv2.erode(mask, kernel, cv2.BORDER_REFLECT)
    mask = np.where(erode >= np.max(erode), 255, 0)
    return mask


if __name__ == "__main__":

    # Load object
    object = bproc.object.create_primitive("MONKEY")
    object.set_cp("category_id", 0)
    object.set_location([0.0, 0.0, 0.7])
    object.set_rotation_euler([-np.pi / 2, 0.0, 0.0])

    # Freeze lights
    light = bproc.types.Light()
    light.set_type("POINT")
    light.set_location([0, 0, 15])
    light.set_energy(3000)

    # Freeze camera location
    bproc.camera.set_resolution(640, 480)
    K = np.array([[638.0, 0.0, 300],
                  [0.0, 637.0, 295],
                  [0.0, 0.0, 1.0]])

    bproc.camera.set_intrinsics_from_K_matrix(K, 640, 480)
    bproc.renderer.enable_depth_output(activate_antialiasing=True)

    poi = bproc.object.compute_poi([object])

    list_of_poses: List[np.ndarray] = []
    for i in range(100):
        # Move and place the camera
        random_transformation = np.random.uniform([-1.5, -1.5, 10], [1.5, 1.5, 15])
        random_rotation = np.random.uniform([0, 0, -np.pi / 3], [0, 0, np.pi / 3])

        cam_pose = bproc.math.build_transformation_mat(random_transformation, random_rotation)
        bproc.camera.add_camera_pose(cam_pose)

        cam_pose = bproc.math.change_source_coordinate_frame_of_transformation_matrix(cam_pose, ["-Y", "X", "-Z"])

        list_of_poses.append(cam_pose)

    data = bproc.renderer.render()
    data.update(bproc.renderer.render_segmap(map_by=["class", "instance", "name"]))

    for i, image in enumerate(data["colors"]):
        cv2.imwrite(f"dataset/{i}_rgb.png", image)

    for i, depth in enumerate(data["depth"]):
        np.save(f"dataset/{i}_depth", depth)

    for i, mask in enumerate(data["instance_segmaps"]):
        mask = np.array(mask)
        cv2.imwrite(f"dataset/{i}_mask.png", generate_erode_mask(mask))

    for i, pose in enumerate(list_of_poses):
        np.savetxt(f"dataset/{i}_pose.txt", np.array(pose))
