from numpy import dtype
import torch
from tqdm import tqdm
import cv2

from slog.keypoint_nets import KeypointNetwork, DataModuleKeypointNet

from slog.utils import convert_tensor_to_cv2, convert_tensor_to_numpy
from slog.renderers import annotate_point_in_image


def compute_spatial_expectations(spatial_probs: torch.Tensor) -> torch.Tensor:

    us = torch.arange(0, spatial_probs.shape[-2], 1, dtype=torch.float32, device=spatial_probs.device)
    vs = torch.arange(0, spatial_probs.shape[-1], 1, dtype=torch.float32, device=spatial_probs.device)
    grid = torch.meshgrid(us, vs, indexing='ij')

    n_keypoints = spatial_probs.shape[1]
    tiled_grid_u = grid[0].unsqueeze(dim=0).tile(n_keypoints, 1, 1)
    tiled_grid_v = grid[1].unsqueeze(dim=0).tile(n_keypoints, 1, 1)

    exp_u = torch.sum(spatial_probs * tiled_grid_u, dim=(-2, -1))
    exp_v = torch.sum(spatial_probs * tiled_grid_v, dim=(-2, -1))

    return torch.stack([exp_u, exp_v]).permute(1, 2, 0)


if __name__ == "__main__":

    # Init. trained module
    model = KeypointNetwork("sandbox/keypointnet/keypointnet-config.yaml")
    trained_model: KeypointNetwork = model.load_from_checkpoint(
        checkpoint_path='/home/kanishk/sereact/slog_models/cap/keypointnet-cap-resnet34-d256-nc8.ckpt',
        yaml_config_path="sandbox/keypointnet/keypointnet-config.yaml"
    )

    trained_model.to("cuda")

    # Init. Datamodule
    data_module = DataModuleKeypointNet("sandbox/keypointnet/keypointnet-config.yaml")
    data_module.prepare_data()
    data_module.setup(stage='fit')

    dataset = data_module.training_dataset

    """

    errors = []
    for i in tqdm(range(100)):
        batch = next(iter(dataset))

        with torch.no_grad():
            spat_exp_a, depth_a = trained_model._forward(batch["RGBs-A"].unsqueeze(dim=0).to("cuda"))
            spat_exp_b, depth_b = trained_model._forward(batch["RGBs-B"].unsqueeze(dim=0).to("cuda"))

        batch_loss = trained_model.loss_function(depth_a,
                                                 depth_b,
                                                 batch["Intrinsics-A"].unsqueeze(dim=0).to("cuda"),
                                                 batch["Intrinsics-B"].unsqueeze(dim=0).to("cuda"),
                                                 batch["Extrinsics-A"].unsqueeze(dim=0).to("cuda"),
                                                 batch["Extrinsics-B"].unsqueeze(dim=0).to("cuda"),
                                                 batch["Masks-A"].unsqueeze(dim=0).to("cuda"),
                                                 batch["Masks-B"].unsqueeze(dim=0).to("cuda"),
                                                 spat_exp_a,
                                                 spat_exp_b)

        errors.append(batch_loss)

    mc = []
    pose = []
    obj = []
    sep = []
    for error in errors:
        mc.append(error["Consistency"])
        pose.append(error["Relative Pose"])
        obj.append(error["Silhoutte"])
        sep.append(error["Separation"])

    print(f"MC: {torch.std_mean(torch.hstack(mc))}")
    print(f"Pose: {torch.std_mean(torch.hstack(pose))}")
    print(f"Obj: {torch.std_mean(torch.hstack(obj))}")
    print(f"Sep: {torch.std_mean(torch.hstack(sep))}")

    """

    batch = next(iter(dataset))

    with torch.no_grad():
        spat_exp_a, depth_a = trained_model._forward(batch["RGBs-A"].unsqueeze(dim=0).to("cuda"))
        spat_exp_b, depth_b = trained_model._forward(batch["RGBs-B"].unsqueeze(dim=0).to("cuda"))

    points = compute_spatial_expectations(spat_exp_a).squeeze(dim=0)

    image = convert_tensor_to_cv2(batch["RGBs-A"].permute(1, 2, 0))
    points = convert_tensor_to_numpy(points.type(torch.int64))

    for point in points:
        image = annotate_point_in_image(image, point[0], point[1], [255, 0, 0], False)

    cv2.imwrite("Image.png", cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
