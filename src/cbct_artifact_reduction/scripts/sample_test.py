import os
from datetime import datetime

import cbct_artifact_reduction.config as cfg
import cbct_artifact_reduction.pigjawdataset as dataset
from cbct_artifact_reduction import lakefs_own
from cbct_artifact_reduction.guided_diffusion import dist_util, logger
from cbct_artifact_reduction.guided_diffusion.script_util import (
    args_to_dict,
    create_model_and_diffusion,
    model_and_diffusion_defaults,
)
from cbct_artifact_reduction.scripts.train_test import create_argparser
from torch.utils.data import DataLoader


def main():
    args = create_argparser().parse_args()
    dist_util.setup_dist()
    logger.configure(os.path.expanduser("~/logs/"))

    today = datetime.now()
    logger.log(f"SAMPLING {today}")
    logger.log(f"args: {args}")
    logger.log("creating model and diffusion...")

    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    logger.log("creating dataloader...")
    client = lakefs_own.CustomBoto3Client(f"{cfg.LAKEFS_DATA_REPOSITORY}")
    inpaintingSliceDataset = dataset.InpaintingSliceDataset(
        client,
        os.path.join(cfg.ROOT_DIR, "training_data.csv"),
        "processed_data/frames/256x256",
        "processed_data/masks/256x256",
    )

    dataloader = DataLoader(inpaintingSliceDataset, batch_size=1, shuffle=True)

    data = iter(dataloader)

    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )

    num_samples = 10

    for i in range(num_samples):
        ground_truth, mask = next(data)
        masked_image = ground_truth * (1 - mask)

        model_kwargs = {}

        sample_fn = diffusion.p_sample_loop_inpainting

        sample, _ = sample_fn(
            model,
            masked_image,
            mask,
            clip_denoised=True,
            model_kwargs=model_kwargs,
        )
        if sample is not None:
            sample = sample.detach().cpu()
            print(sample.shape)

    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()
    pass


if __name__ == "__main__":
    main()
