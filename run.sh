#!/bin/bash
# Run this from the cbct-artifact-reduction folder


if [[ -z "$1" ]]; then
    echo "Usage: ./run train|sample|resume"
    exit 1
elif [[ "$1" == "sample" ]]; then
    echo "Input: sample. OK."
elif [[ "$1" == "train" ]]; then
    echo "Input: train. OK."
elif [[ "$1" == "resume" ]]; then
    echo "Input: resume. OK."
else
    echo "Usage: ./run train|sample|resume"
    exit 1
    echo "Invalid Input. Leaving..."
    exit 1
fi

# GENERAL VARIABLES
LOG_DIR="~/logs"

# MODEL FLAGS
MODEL_FLAGS="--image_size 256 --num_channels 128 --class_cond False --num_res_blocks 2 --num_heads 1 --learn_sigma True --use_scale_shift_norm False --attention_resolutions 16"
DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule linear --rescale_learned_sigmas False --rescale_timesteps False"
TRAIN_FLAGS="--lr 1e-4 --batch_size 8"

# TRAINING VARIABLES
TRAIN_SCRIPT="src/cbct_artifact_reduction/scripts/train_test.py"

# SAMPLING VARIABLES
SAMPLE_SCRIPT="src/cbct_artifact_reduction/scripts/sample_test.py"
SAMPLE_FLAGS="--random_masks False"

MODEL_PATH="/home/julian.bopp/logs/model095000.pt"

# PRINTING VARIABLES
echo "Logging to: $LOG_DIR"

echo "--------------------"
echo "MODEL_FLAGS:"
echo $MODEL_FLAGS
echo "--------------------"
echo "DIFFUSION_FLAGS:"
echo $DIFFUSION_FLAGS
echo "--------------------"
echo "TRAIN_FLAGS:"
echo $TRAIN_FLAGS
echo "--------------------"
echo "SAMPLE_FLAGS:"
echo $SAMPLE_FLAGS
echo "--------------------"
echo "TRAIN_FLAGS:"

if [[ "$1" == "train" ]]; then
    echo "Starting training"
    echo "Running $TRAIN_SCRIPT"
    python $TRAIN_SCRIPT $TRAIN_FLAGS $MODEL_FLAGS $DIFFUSION_FLAGS
elif [[ "$1" == "resume" ]]; then
    echo "Resume training"
    echo "Using model: $MODEL_PATH"
    echo "Running $TRAIN_SCRIPT"
    python $TRAIN_SCRIPT --resume_checkpoint $MODEL_PATH $TRAIN_FLAGS $MODEL_FLAGS $DIFFUSION_FLAGS
elif [[ "$1" == "sample" ]]; then
    echo "Starting sampling"
    echo "Running $SAMPLE_SCRIPT"
    python $SAMPLE_SCRIPT --model_path $MODEL_PATH $MODEL_FLAGS $DIFFUSION_FLAGS
fi
