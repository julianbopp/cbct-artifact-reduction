import argparse

import cbct_artifact_reduction.guided_diffusion.script_util as script_util


def create_train_argparser():
    defaults = dict(
        data_dir="",
        log_dir="~/logs/",
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
        num_channels=128,
        class_cond=False,
        num_res_blocks=2,
        num_heads=1,
        learn_sigma=True,
        use_scale_shift_norm=False,
        attention_resolutions=16,
        diffusion_steps=1000,
        noise_schedule="linear",
        rescale_learned_sigmas=False,
        rescale_timesteps=False,
        data_csv="training_data.csv",
    )
    defaults.update(script_util.model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    script_util.add_dict_to_argparser(parser, defaults)
    return parser


def create_sample_argparser():
    defaults = dict(
        data_dir="",
        log_dir="~/samples/",
        # gt_dir="",
        adapted_samples="",
        sub_batch=16,
        clip_denoised=True,
        num_samples=1,
        batch_size=1,
        use_ddim=False,
        model_path="",
        num_ensemble=1,
        image_size=256,
        random_masks=True,
        data_csv="sample_data.csv",
        return_item_info=True,
        lakefs_folder="processed_data/frames/256x256",
    )
    defaults.update(script_util.model_and_diffusion_defaults())  # type: ignore
    parser = argparse.ArgumentParser()
    script_util.add_dict_to_argparser(parser, defaults)
    return parser
