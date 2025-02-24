import argparse
import os
from datetime import datetime

import cbct_artifact_reduction.config as cfg
import cbct_artifact_reduction.lakefs_own as lakefs_own
import cbct_artifact_reduction.pigjawdataset as dataset
from cbct_artifact_reduction.guided_diffusion import dist_util, logger
from cbct_artifact_reduction.guided_diffusion.resample import (
    create_named_schedule_sampler,
)
from cbct_artifact_reduction.guided_diffusion.script_util import (
    add_dict_to_argparser,
    args_to_dict,
    create_model_and_diffusion,
    model_and_diffusion_defaults,
)
from cbct_artifact_reduction.guided_diffusion.train_util import TrainLoop
from torch.utils.data import DataLoader


def main():
    args = create_argparser().parse_args()
    dist_util.setup_dist()
    logger.configure(os.path.expanduser("~/logs/"))

    today = datetime.now()
    logger.log(f"TRAINING {today}")
    logger.log(f"args: {args}")
    logger.log("creating model and diffusion...")

    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )

    model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(
        args.schedule_sampler,
        diffusion,
    )

    logger.log("creating dataloader...")
    client = lakefs_own.CustomBoto3Client(f"{cfg.LAKEFS_DATA_REPOSITORY}")
    inpaintingSliceDataset = dataset.InpaintingSliceDataset(
        client,
        os.path.join(cfg.ROOT_DIR, args.data_csv),
        "processed_data/frames/256x256",
        random_masks=args.random_masks,
    )

    num_epochs = args.num_epochs
    logger.log("training...")
    step = 0
    for epoch in range(num_epochs):
        logger.log(f"epoch {epoch + 1}/{num_epochs}")
        dataloader = DataLoader(
            inpaintingSliceDataset, batch_size=args.batch_size, shuffle=True
        )
        data = iter(dataloader)

        step = TrainLoop(
            model=model,
            diffusion=diffusion,
            data=data,
            batch_size=args.batch_size,
            microbatch=args.microbatch,
            lr=args.lr,
            ema_rate=args.ema_rate,
            log_interval=args.log_interval,
            save_interval=args.save_interval,
            resume_checkpoint=args.resume_checkpoint,
            use_fp16=args.use_fp16,
            fp16_scale_growth=args.fp16_scale_growth,
            schedule_sampler=schedule_sampler,
            weight_decay=args.weight_decay,
            lr_anneal_steps=args.lr_anneal_steps,
            step=step if step else 0,
        ).run_loop()
        # Make sure to not resume checkpoint again after first epoch:
        args.resume_checkpoint = ""


def create_argparser():
    defaults = dict(
        data_dir="",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=1000,
        save_interval=5000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        random_masks=True,
        num_epochs=10000,
        data_csv="training_data.csv",
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
