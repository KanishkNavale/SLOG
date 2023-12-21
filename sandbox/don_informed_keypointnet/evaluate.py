import torch
from tqdm import tqdm

from slog.ende_zu_ende.don_informed_keypointnet import DONInformedKeypointNet, DataModuleDONInformedKeypointNet


if __name__ == "__main__":
    # Init. trained module
    yaml_path = "sandbox/don_informed_keypointnet/dik-config.yaml"
    model = DONInformedKeypointNet(yaml_path)
    trained_model: DONInformedKeypointNet = model.load_from_checkpoint(
        checkpoint_path='/home/kanishk/sereact/slog_models/don_informed_keypointnet/DIK-don-resnet_18-d3-pixelwise_correspondence_loss-keypointnet-resnet_34-d256-nc8.ckpt',
        yaml_config_path=yaml_path)

    trained_model.to("cuda")

    # Init. Datamodule
    data_module = DataModuleDONInformedKeypointNet(yaml_path)
    data_module.prepare_data()
    data_module.setup(stage='fit')

    dataset = data_module.training_dataset

    errors = []
    for i in tqdm(range(100)):
        batch = next(iter(dataset))

        with torch.no_grad():
            desc_a = trained_model._forward_don(batch["RGBs-A"].unsqueeze(dim=0).to("cuda"))
            desc_b = trained_model._forward_don(batch["RGBs-B"].unsqueeze(dim=0).to("cuda"))

            spat_exp_a, depth_a = trained_model._forward_keypointnet(desc_a)
            spat_exp_b, depth_b = trained_model._forward_keypointnet(desc_b)

            print(desc_a[:, ])

        batch_loss = trained_model.keypointnet_loss(depth_a,
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
