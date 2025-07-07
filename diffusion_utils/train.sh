#!/bin/bash
#BSUB -n 1
#BSUB -o train.out
#BSUB -R rusage[mem=20G]
#BSUB -gpu num=1:j_exclusive=yes

#  Copyright 2025, Bo Wang Lab
# 
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
 
#  http://www.apache.org/licenses/LICENSE-2.0
 
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
# Based on code from https://github.com/bowang-lab/MorphoDiff

echo $(nvcc --version)$
echo which nvcc: $(which nvcc)

# Echo the path to the Python interpreter
echo "Using Python from: $(which python)"
echo "CUDA is available: $(python -c 'import torch; print(torch.cuda.is_available())')"


## Local path to the cloned Github repository
export ROOT="/home/lohmanf1/stylegan-t"

## Fixed parameters ##
export CKPT_NUMBER=0
export TRAINED_STEPS=0

## set the path to the pretrained VAE model. Downloaded from: https://huggingface.co/CompVis/stable-diffusion-v1-4 
export VAE_DIR="CompVis/stable-diffusion-v1-4"

## set the path to the log directory
export LOG_DIR="${ROOT}/results/diffusion/"
## check if LOG_DIR exists, if not create it
if [ ! -d "$LOG_DIR" ]; then
  mkdir -p $LOG_DIR
fi

## set the experiment name
export EXPERIMENT="TENX146_guidance"

## set the path to the pretrained model, which could be either pretrained Stable Diffusion, or a pretrained MorphoDiff model
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
## Feature extractor config
export PROC_CFG="${ROOT}/diffusion_utils/preprocessor_config.json" 

## set the path to the training image folder and precomputed embedding file
export TRAIN_DIR="${ROOT}/data/patches_55um_paired/TENX146/"
export TRAIN_EMBS="${ROOT}/data/space_ranger_outputs/TENX146/gene_clip_emb_146.npy"

## set the path to the checkpointing log file in .csv format. Should change the MorphoDiff to SD if training unconditional Stable Diffusion 
export CKPT_LOG_FILE="${LOG_DIR}${EXPERIMENT}_log/${EXPERIMENT}_MorphoDiff_checkpoints.csv"

## set the validation prompts/perturbation ids, separated by ,
export VALID_PROMPT="${ROOT}/data/embeddings/validation_prompts_clip.npy" 

## the header for the checkpointing log file
export HEADER="dataset_id,log_dir,pretrained_model_dir,checkpoint_dir,seed,trained_steps,checkpoint_number"
mkdir -p ${LOG_DIR}${EXPERIMENT}_log

## embedding of the empty prompt for classifier-free guidance
export UNCOND="${ROOT}/data/embeddings/clip_unconditional_embedding.npy"
## guidance scale for validation inference
export GUIDE=3

## Function to get column index by header name
get_column_index() {
    local header_line=$1
    local column_name=$2
    echo $(echo "$header_line" | tr ',' '\n' | nl -v 0 | grep "$column_name" | awk '{print $1}')
}

# Check if the checkpointing log CSV file exists
if [ ! -f "$CKPT_LOG_FILE" ]; then
    # If the file does not exist, create it and add the header
    echo "$HEADER" > "$CKPT_LOG_FILE"
    echo "CSV checkpointing log file created with header: $HEADER"

elif [ $(wc -l < "$CKPT_LOG_FILE") -eq 1 ]; then
    # overwrite the header line
    echo "$HEADER" > "$CKPT_LOG_FILE"
    echo "CSV checkpointing log file header overwritten with: $HEADER"

else
    echo "CSV checkpointing log file exists in $CKPT_LOG_FILE"
    echo "Reading the last line of the log file to resume training"
    # If the file exists, read the last line
    LAST_LINE=$(tail -n 1 "$CKPT_LOG_FILE")
    
    # Extract the header line to determine the index of "checkpoint_dir" column
    HEADER_LINE=$(head -n 1 "$CKPT_LOG_FILE")
    CHECKPOINT_DIR_INDEX=$(get_column_index "$HEADER_LINE" "checkpoint_dir")

    # Extract the checkpoint_dir value from the last line
    MODEL_NAME=$(echo "$LAST_LINE" | cut -d',' -f$(($CHECKPOINT_DIR_INDEX + 1)))

    # Extract the last column from the last line
    LAST_COLUMN=$(echo "$LAST_LINE" | awk -F',' '{print $NF}')
    # Convert the last column to an integer
    CKPT_NUMBER=$((LAST_COLUMN))

    # get the number of trained steps so far
    TRAINED_STEPS_INDEX=$(get_column_index "$HEADER_LINE" "trained_steps")
    TRAINED_STEPS=$(echo "$LAST_LINE" | cut -d',' -f$(($TRAINED_STEPS_INDEX + 1)))

fi

# add 1 to the value of CKPT_NUMBER
export CKPT_NUMBER=$((${CKPT_NUMBER}+1))
export OUTPUT_DIR=${LOG_DIR}${EXPERIMENT}_log

echo "Checkpoint number: $CKPT_NUMBER"
echo "Model directory: $MODEL_NAME"
echo "Output directory: $OUTPUT_DIR"
echo "Data directory: $TRAIN_DIR"
echo "Trained steps: $TRAINED_STEPS"


accelerate launch --mixed_precision="fp16" ../train_diffusion.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --preprocessor_config_file=$PROC_CFG \
  --train_data_dir=$TRAIN_DIR \
  --dataset_id=$EXPERIMENT \
  --dataset_embs=$TRAIN_EMBS \
  --enable_xformers_memory_efficient_attention \
  --resolution=128 \
  --random_flip \
  --use_ema \
  --train_batch_size=32 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --learning_rate=1e-05 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --validation_steps=1 \
  --validation_prompts=$VALID_PROMPT  \
  --checkpointing_steps=1 \
  --output_dir=$OUTPUT_DIR \
  --image_column="image" \
  --caption_column="gex_emb" \
  --pretrained_vae_path=$VAE_DIR \
  --cache_dir="/tmp/" \
  --report_to="wandb" \
  --logging_dir="${LOG_DIR}${EXPERIMENT}_log" \
  --seed=42 \
  --checkpointing_log_file=$CKPT_LOG_FILE \
  --checkpoint_number=$CKPT_NUMBER \
  --checkpoints_total_limit=5 \
  --trained_steps=$TRAINED_STEPS \
  --tracker_project_name=diffusion \
  --num_train_epochs=15 \
  --classifier_free_guidance \
  --uncond_embedding=$UNCOND \
  --guidance_scale=$GUIDE \
  --p_uncond=0.1 
