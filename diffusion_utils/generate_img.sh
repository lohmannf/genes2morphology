#!/bin/bash
#BSUB -gpu num=1
#BSUB -R rusage[mem=5G]
#BSUB -n 1
#BSUB -o generation.out

#  Copyright 2025, Bo Wang Lab
#  Modifications Copyright 2025 F. Hoffmann-La Roche AG
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


## Define/adjust the parameters ##

## Guidance scale according to Imagen definition
GUIDE=3

## CKPT_PATH is the path to the checkpoint folder. 
CKPT_PATH="/home/lohmanf1/scratch/diffusion/TENX146_log/checkpoint-132275/"

## Set path to the directory where you want to save the generated images#
GEN_IMG_PATH="../quality_metrics/diff_guide3/"
## Set the number of images you want to generate (per prompt)
NUM_GEN_IMG=4

## File with prompt embeddings and dummy prompt for CFG training
PROMPT_FILE="../data/embeddings/gene_clip_emb_sample1-5.npy" 
UNCOND="../data/embeddings/clip_unconditional_embedding.npy"

## Generate images
python generate_img.py \
--model_checkpoint $CKPT_PATH \
--vae_path $CKPT_PATH \
--prompt_file $PROMPT_FILE \
--output_type "pil" \
--gen_img_path $GEN_IMG_PATH \
--num_imgs $NUM_GEN_IMG \
--classifier_free_guidance \
--guidance_scale $GUIDE \
--uncond_embedding $UNCOND \
--same_seed

# only use classifier-free guidance if model was trained with p_uncond > 0!
# inference with guidance will still run otherwise but results make no sense