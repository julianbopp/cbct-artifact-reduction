"""
Train a diffusion model on images.
"""

import argparse
import sys

sys.path.append("..")
sys.path.append(".")
from datetime import datetime

import torch as th
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
from cbct_artifact_reduction.pigjawdataset import InpaintingSliceDataset

# viz = Visdom(port=8851)


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()
    today = datetime.now()
    # print('today', today)
    logger.log("TRAINING " + str(today))
    logger.log("args: " + str(args))
    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )

    print("args", args)

    # breakpoint()

    model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(
        args.schedule_sampler, diffusion, maxt=1000
    )

    # print('SEGMENTATION TRAIN MAIN')
    logger.log("creating data loader...")
    ds = InpaintingSliceDataset(args.data_dir, test_flag=False)
    print("ds", ds)
    datal = th.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=True)
    data = iter(datal)

    # print('data', data)

    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        classifier=None,
        data=data,
        dataloader=datal,
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
    ).run_loop()


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
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
