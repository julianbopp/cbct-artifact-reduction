import os
import time

import cbct_artifact_reduction.argparser_config as argparser_config
import cbct_artifact_reduction.model_util as model_util
import cbct_artifact_reduction.pigjawdataset as dataset
import nibabel as nib
import torch
from cbct_artifact_reduction.csvcreator import (
    get_random_entries_by_scanner_fov_mandible,
    get_random_slice_from_id,
)
from cbct_artifact_reduction.guided_diffusion import dist_util, logger
from cbct_artifact_reduction.guided_diffusion.script_util import (
    args_to_dict,
    create_model_and_diffusion,
    model_and_diffusion_defaults,
)
from filelock import FileLock
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

args = argparser_config.create_sample_argparser().parse_args()


fixed_ids = get_random_entries_by_scanner_fov_mandible(
    "data.csv",
    1,
    exclude_mandibles=[2, 3, 4, 5, 10, 7, 9, 12],
).id.tolist()

fixed_filenames = get_random_slice_from_id("data.csv", fixed_ids)

number_of_samples_per_slice = 5

fixed_filenames = [
    filename for filename in fixed_filenames for _ in range(number_of_samples_per_slice)
]

logger.configure(os.path.expanduser(args.log_dir))


def create_dataloader():
    validation_dataloader = model_util.create_dataloader(
        filenames=fixed_filenames,
        random_masks=False,
        lakefs_folder=args.lakefs_folder,
        batch_size=1,
        shuffle=False,
    )
    return validation_dataloader


CHECKPOINT_DIR = os.path.expanduser("~/logs/")  # Directory containing checkpoints
OUTPUT_DIR = os.path.expanduser("~/samples/")  # Directory to save outputs


def sample_model(checkpoint_path):
    lock = FileLock("/tmp/julian.bopp.model_sampling.lock")  # Create a lock file

    with lock:  # This will block until the lock is available
        logger.info(f"Loading model from: {checkpoint_path}")

        DEVICE = "cpu"

        if torch.cuda.is_available():
            DEVICE = f"cuda:{model_util.get_free_gpu()}"

        logger.info(f"Using DEVICE: {DEVICE}")

        model, diffusion = create_model_and_diffusion(
            **args_to_dict(args, model_and_diffusion_defaults().keys())
        )
        try:
            model.load_state_dict(
                dist_util.load_state_dict(checkpoint_path, map_location=DEVICE)
            )
        except:
            logger.error("Could not load model. Trying to load again on CPU.")
            model.load_state_dict(
                dist_util.load_state_dict(
                    checkpoint_path, map_location=torch.device("cpu")
                )
            )
        model.eval()
        logger.info("Model loaded")

        data = create_dataloader()
        logger.info("Dataloader created")

        if args.use_fp16:
            model.convert_to_fp16()

        # Create output directory OUTPUTDIR/checkpointname/
        model_output_dir = os.path.join(OUTPUT_DIR, os.path.basename(checkpoint_path))

        if not os.path.exists(model_output_dir):
            os.mkdir(model_output_dir)
        logger.info("Output directory created at {}".format(model_output_dir))

        n = len(data)

        with torch.no_grad():
            for i, item in enumerate(data):
                ground_truth, mask, info = item["slice"], item["mask"], item["info"]
                masked_image = ground_truth * (1 - mask)
                model_kwargs = {}

                sample_fn = diffusion.p_sample_loop_inpainting

                slice_name = info["slice_name"]
                logger.log(f"Generating sample_{slice_name} ({i}/{n})")
                sample, _ = sample_fn(
                    model,
                    masked_image,
                    mask,
                    clip_denoised=args.clip_denoised,
                    model_kwargs=model_kwargs,
                )
                if sample is not None:
                    sample = sample.detach().cpu()
                    mask = mask.detach().cpu()
                    ground_truth = ground_truth.detach().cpu()
                    sample = torch.cat([sample, ground_truth, mask], dim=1).numpy()
                    sample_nifti_object = nib.nifti1.Nifti1Image(sample, None)

                    if args.batch_size == 1:
                        info = dataset.clean_dict(info)
                    nib.save(
                        sample_nifti_object,
                        os.path.join(OUTPUT_DIR, f"sample_{slice_name}"),
                    )

                    logger.log(f"Saved sample_{slice_name} ({i}/{n})")
                    logger.log(info)

        logger.info("Freeing GPU memory")
        del model
        del diffusion
        torch.cuda.empty_cache()
        logger.info(f"Finished processing {checkpoint_path}")


class CheckpointHandler(FileSystemEventHandler):
    def on_modified(self, event):
        if isinstance(event.src_path, str):
            if event.src_path.endswith(".pt") and "opt" not in event.src_path:
                # src_path is a path to a modelXXXXX.pt or ema_X.XXXX_XXXX.pt file
                time.sleep(60)  # Small delay to ensure the file is fully written
                sample_model(event.src_path)


def monitor_checkpoints():
    event_handler = CheckpointHandler()
    observer = Observer()
    observer.schedule(event_handler, CHECKPOINT_DIR, recursive=False)
    observer.start()

    try:
        logger.log(f"Watching {CHECKPOINT_DIR} for new checkpoints...")
        while True:
            time.sleep(5)
    except KeyboardInterrupt:
        observer.stop()

    observer.join()


if __name__ == "__main__":
    args = argparser_config.create_sample_argparser().parse_args()
    monitor_checkpoints()
