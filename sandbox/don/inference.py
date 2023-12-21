
import cv2
import numpy as np
import torch
from sklearn.decomposition import PCA

from slog.datatypes import VisionInputData
from slog.utils import convert_tensor_to_cv2, convert_tensor_to_numpy
from slog.don import DON

from slog.keypoint_apps.pose_graph_generator import PoseGraphGenerator

from torchvision.transforms import Compose, Resize, Normalize


if __name__ == "__main__":

    model = DON("sandbox/don/don-config.yaml")
    model = model.load_from_checkpoint(
        checkpoint_path='~/sereact/slog_models/cap/don-d3-resnet50-nc512-correspondence.ckpt',
        yaml_config_path="sandbox/don/don-config.yaml"
    )
    model.to("cuda")

    image_a = torch.as_tensor(cv2.imread("dataset/multi_cap_dataset/rgbs/10.jpg") / 255, dtype=torch.float32)
    image_b = torch.as_tensor(cv2.imread("dataset/multi_cap_dataset/rgbs/11.png") / 255, dtype=torch.float32)

    depth_a = torch.ones_like(image_a).to("cuda")
    intrinsic = torch.as_tensor(np.loadtxt("dataset/single_object_dataset/dataset/intrinsics.txt"), dtype=torch.float32).to("cuda")

    desc_a = model._forward(image_a.permute(2, 0, 1).unsqueeze(0).to("cuda")).squeeze(0).permute(1, 2, 0)
    desc_b = model._forward(image_b.permute(2, 0, 1).unsqueeze(0).to("cuda")).squeeze(0).permute(1, 2, 0)

    dim_reducer = PCA(n_components=3)
    _desc_a = desc_a.reshape(desc_a.shape[0] * desc_a.shape[1], desc_a.shape[2])
    _desc_b = desc_b.reshape(desc_b.shape[0] * desc_b.shape[1], desc_b.shape[2])

    _desc_a: np.ndarray = convert_tensor_to_numpy(_desc_a)
    _desc_b: np.ndarray = convert_tensor_to_numpy(_desc_b)

    _desc_a = dim_reducer.fit_transform(_desc_a)
    _desc_b = dim_reducer.transform(_desc_b)

    _desc_a = _desc_a.reshape(desc_a.shape[0], desc_a.shape[1], 3)
    _desc_b = _desc_b.reshape(desc_b.shape[0], desc_b.shape[1], 3)

    images = np.hstack([convert_tensor_to_cv2(image_a), convert_tensor_to_cv2(image_b)])
    desc = np.hstack([convert_tensor_to_cv2(_desc_a)])

    cv2.imwrite("debug.png", desc)

    vis_data = VisionInputData(rgb=images, depth=depth_a,
                               intrinsic=intrinsic, extrinsic=torch.eye(4).to("cuda"), descriptor=torch.hstack([desc_a, desc_b]).to("cuda"))

    app = PoseGraphGenerator(vis_data)
    app.run()
