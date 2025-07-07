# Stable Diffusion #
## Setup ##
The code for the Stable Diffusion model is independent from the rest of the codebase.
Please follow the instructions from https://github.com/bowang-lab/MorphoDiff for setting up your environment.
Before installing the custom diffusers library, replace `MorphoDiff/morphodiff/diffusers/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py` by the version provided in this directory to enable classifier-free guidance.

## Data Preparation ##
Finetuning of Stable Diffusion can be performed on an image folder of preprocessed patches obtained with `../dump_patches.py`.
Run the following command to add a `metadata.jsonl` file that is required for compatibility with HuggingFace `datasets`
```
python dump_metadata.py --image-dir <IMAGE FOLDER>
```
Gene expression embeddings of choice (e.g. CLIP, scGPT) can be used as condition.
Note that in the gene expression embedding file, the order of the spots has to be identical to the lexicographical order of the image patch file names in your image folder.
Consult `../analysis_notebooks/scGPT_embeddings.ipynb` for a tutorial on how to extract spot embeddings with scGPT Continual Pretrained.

## Finetuning ##
We provide a script for running the finetuning of the natural image - pretrained Stable Diffusion Pipeline on custom prompt embeddings and images.
```
bash train.sh
```
If you want to jointly train an unconditional model to perform classifier-free guidance at inference time, a dummy prompt that symbolizes the absence of a condition
has to be provided. We recommend using your prompt encoding model of choice to embed the vector of all-zeroes as a dummy prompt.

## Generating Morphologies ##
```
bash generate_img.sh
```
To perform prompt interpolation, you can use `../misc/interpolate_prompts.py` to obtain spherically interpolated embeddings between two promtps.
