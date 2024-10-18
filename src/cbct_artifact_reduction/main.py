import platform
import os


from omegaconf import OmegaConf
import torch
from torch.utils.data import DataLoader
import numpy as np

from cbct_artifact_reduction.data.lakefs import Boto3Client
from cbct_artifact_reduction.data import BratsSliceDataset

import monai


def train_model(cfg):
    # set the process title if the OS is not MAC
    if platform.system() != "Darwin":
        import setproctitle

        tokens = os.getlogin().split(".")
        initials = [x[0] for x in tokens]
        setproctitle.setproctitle(
            "DV - train" + " User: (" + "".join(initials).upper() + ")"
        )

    # print configuration of monai
    monai.config.print_config()

    # define the device for the computation
    if cfg.training.device >= 0:
        device = torch.device("cuda:" + str(cfg.training.device))
    else:
        device = torch.device("cpu")

    # define the dataset and the data loader
    data_loader_lakefs = Boto3Client(
        endpoint=cfg.lakefs.s3_endpoint,
        access_key=cfg.lakefs.access_key,
        secret_key=cfg.lakefs.secret_key,
        repo_name=cfg.lakefs.data_repository,
        commit_id=cfg.lakefs.commit,
        local_cache_path=cfg.lakefs.cache_path,
        ca_path=cfg.lakefs.ca_path,
    )

    # define the training set
    data_set_training = BratsSliceDataset(
        data_loader_lakefs, cfg.training.cache_files, cfg.training.modalities
    )

    # define the model
    model = monai.networks.nets.UNet(
        spatial_dims=cfg.model.spatial_dims,
        in_channels=len(cfg.training.modalities),
        out_channels=cfg.model.out_channels,
        channels=cfg.model.channels,
        strides=cfg.model.strides,
        num_res_units=cfg.model.num_res_units,
    ).to(device)
    model.train()

    # define the data loader
    data_loader = DataLoader(
        data_set_training,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.training.num_workers,
        shuffle=True,
        pin_memory=True,
        persistent_workers=True,
    )

    # define the optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.training.lr_rate)

    # define the loss function
    loss_function = monai.losses.DiceLoss(
        include_background=False, softmax=True, to_onehot_y=True
    )

    ce_loss = torch.nn.CrossEntropyLoss()

    # start the training
    for epoch_idx in range(1, cfg.training.num_epoch + 1):
        step = 0
        for run_idx, data in enumerate(data_loader):

            step += 1

            inputs, labels = data["images"].to(device), data["labels"].to(device)

            # # set all previous gradients of the model to zero
            # # this is faster the model.zero_grad
            for param in model.parameters():
                param.grad = None

            outputs = model(inputs)
            dice_loss_values = (
                loss_function(outputs, labels) * cfg.training.DICE_loss_weight
            )
            ce_loss_value = (
                ce_loss(outputs, labels.squeeze().long()) * cfg.training.CE_loss_weight
            )
            loss = dice_loss_values + ce_loss_value

            loss.backward()
            optimizer.step()

            epoch_len = len(data_set_training) // data_loader.batch_size
            print(
                f"Epoch: {epoch_idx} Run: {step}/{epoch_len}, DICE: {dice_loss_values.item():.4f}, CE Loss {ce_loss_value:.4f},"
                f" Total:{loss:.4f}"
            )


def main() -> None:

    # get the cli commands
    cli_conf = OmegaConf.from_cli()
    # load the config file
    if cli_conf.config is not None:
        cfg_conf = OmegaConf.load(cli_conf.config)
        # merge the commands
        cfg = OmegaConf.merge(cfg_conf, cli_conf)
    else:
        cfg = cli_conf

    # set the seeds if needed for torch and numpy
    if cfg.seeds.torch != -1:
        torch.manual_seed(cfg.seeds.torch)

    if cfg.seeds.numpy != -1:
        np.random.seed(cfg.seeds.numpy)

    if cfg.mode == "train":
        print("-------------------------")
        print("PERFORMING MODEL TRAINING")
        print("-------------------------")
        train_model(cfg)
    elif cfg.mode == "infer":
        print("Performing model inference")
        # inference(cfg)
    else:
        raise KeyError(" ".join(("Mode", cfg.mode, "is not valid operation mode")))


if __name__ == "__main__":
    main()
