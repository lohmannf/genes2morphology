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
import os
import torch_fidelity
from collections import OrderedDict
import pandas as pd
import torch
import random
import torchvision.io as io
import numpy as np
import click 

def set_seed(seed):
    """
    Set seed for reproducibility.

    Parameters
    ----------
    seed: int
        The seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    return

class SimpleDataset(torch.utils.data.Dataset):
    """
    Simple tensor dataset that returns batches as raw tensors
    """
    def __init__(self, tensors) -> None:
        self.tensors = tensors

    def __getitem__(self, index):
        return self.tensors[index]

    def __len__(self):
        return self.tensors.size(0)


def calculate_metrics(ref_path, gen_path, result_path, experiment):
    """
    Calculate FID and KID for the generated images and write/append the results to a csv file

    Parameters
    ----------
    ref_path: list 
        Path(s) to the reference images

    gen_path: str 
        Path to the generated images

    result_path: str
        Path to directory with results file
    
    experiment: str 
        Experiment name

    Returns
    -------
    None
    """
    print('Calculating metrics')
    print('Reference path(s): ', ref_path)
    print('Model generated path: ', gen_path)
    
    metric = OrderedDict()
    metric['experiment'] = []
    metric['FID'] = []
    metric['KID_mean'] = []
    metric['KID_std'] = []
    metric['seed'] = []

    seed = 42
    set_seed(seed)
    ref_imgs = []
    model_imgs = []
    
    ref_img_paths = np.concatenate([np.array([os.path.join(p, f) for f in os.listdir(p)]) for p in ref_path])
    gen_img_paths = [os.path.join(gen_path, p) for p in os.listdir(gen_path)]
    if len(ref_img_paths) != len(gen_img_paths):
        random.shuffle(ref_img_paths)
        random.shuffle(gen_img_paths)
        min_sz = min(len(ref_img_paths), len(gen_img_paths))
        ref_img_paths = ref_img_paths[:min_sz]
        gen_img_paths = gen_img_paths[:min_sz]

    for img_path in ref_img_paths:
        img = io.read_image(img_path)
        ref_imgs.append(img)

    for img in gen_img_paths:
        if img.endswith(".png"):
            img = io.read_image(img)
            model_imgs.append(img)
    ref_imgs = torch.stack(ref_imgs).to(torch.uint8)
    model_imgs = torch.stack(model_imgs).to(torch.uint8)

    model_imgs = SimpleDataset(model_imgs)
    ref_imgs = SimpleDataset(ref_imgs)
    # calculate FID, KID
    set_seed(seed)
    metrics_dict = torch_fidelity.calculate_metrics(
        input1=model_imgs,
        input2=ref_imgs,
        cuda=True if torch.cuda.is_available() else False,
        fid=True,
        kid=True,
        batch_size=256,
        kid_subset_size=500,
        verbose=True,
    )

    metric['experiment'].append(experiment)
    metric['FID'].append(metrics_dict['frechet_inception_distance'])
    metric['KID_mean'].append(
        metrics_dict['kernel_inception_distance_mean'])
    metric['KID_std'].append(
        metrics_dict['kernel_inception_distance_std'])
    metric['seed'].append(seed)
    metric_df = pd.DataFrame(metric)
    
    print('Successfully calculated metrics')
    print('--'*50)
    print()

    with open(result_path, 'a') as f:
        metric_df.to_csv(f, header=f.tell() == 0, index=False)
    return


def create_result_file(result_path):
    """
    Create a result file to save the image quality metrics.
    """
    result_path = os.path.join(result_path, "image_quality_metrics.csv")

    if not os.path.exists(result_path):
        with open(result_path, 'w') as f:
            f.write('experiment,FID,' +
                    'KID_mean,KID_std,seed\n')

    return result_path

@click.command("cli", context_settings={"show_default": True})
@click.option("--experiment", type=str, help="Experiment name to display in results file")
@click.option("--ref-img-path", type=str, help="Folder(s) with reference images as comma separated list, e.g. dir1,dir2")
@click.option("--gen-img-path", type=str, help="Folder with generated images")
@click.option("--out-dir", type=str, help="Directory for output file", default="./")
def main(experiment,
         ref_img_path,
         gen_img_path,
         out_dir):

    ref_img_paths = ref_img_path.split(",")

    os.makedirs(out_dir, exist_ok=True)
    result_path = create_result_file(out_dir)
    
    calculate_metrics(
        ref_img_paths, gen_img_path, result_path, experiment)

if __name__ == "__main__":
    main()