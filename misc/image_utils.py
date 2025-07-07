# Copyright (c) 2025, F. Hoffmann-La Roche Ltd.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from PIL import Image
import numpy as np
import scanpy as sc
import torch
import click
import glob

def images2gif(img_dir: str, out_file: str, frame_duration: int = 100, n_dup: int = 1):
    """
    Create a gif from a folder of lexicographically ordered images

    Parameters
    ----------
    img_dir: str
        Image folder path

    out_file: str
        Path to output file

    frame_duration: int
        How long to show each frame, default = 100

    n_dup: int
        How often to duplicate the first and last frame to increase their relative duration,
        default = 1

    Returns
    -------
    None
    """
    images = []
    # get all the images in the folder
    for filename in sorted(glob.glob(f'{img_dir}/*.png')): # loop through all png files in the folder
        im = Image.open(filename)
        images.append(im) 

    # calculate the frame number of the last frame (ie the number of images)
    last_frame = (len(images)) 

    # create extra copies of first and last frame
    im_last = images[last_frame-1]
    im_first = images[0]
    for _ in range(0, n_dup):
        images.append(im_last)
        images.insert(0, im_first)

    # save as a gif   
    images[0].save(out_file, save_all=True, append_images=images[1:], optimize=False, duration=frame_duration, loop=0)


def choose_samples(dir: str, n_samples: int):
    """
    Select n_samples random files from dir
    """
    files = np.array(sorted([f for f in os.listdir(dir) if f.endswith(".png")]))
    msk = np.random.choice(len(files), size=n_samples, replace=False)
    files  = files[msk]
    files = [os.path.join(dir, f) for f in files]
    return files, msk


def image_grid(imgs: list, outfile: str, n_row: int, n_col: int):
    """
    Create a grid from a list of image files.
    Files are arranged in row-major order.

    Parameters
    ----------
    imgs: list
        Image filenames, `len(imgs)` has to be equal to `n_row*n_col`

    outfile: str
        Path to the output file with image grid

    n_row: int
        Number of rows in grid

    n_col: int
        Number of columns in grid

    Returns
    --------
    None
    """
    if isinstance(imgs[0], str):
        imgs = [np.array(Image.open(f)) for f in imgs]
    height, width = imgs[0].shape[:-1]

    im_out = np.stack(imgs, axis = 0).reshape(n_row, n_col, height, width, 3)
    im_out = np.concatenate(im_out, axis = 1)
    im_out = np.concatenate(im_out, axis = 1)

    os.makedirs(os.path.dirname(outfile), exist_ok = True)
    im_out = Image.fromarray(im_out)
    im_out.save(outfile)


def get_prompt(msk, adata):
    """
    Extract gene expression prompts corresponding to boolean or index mask

    Parameters
    ----------
    msk: np.ndarray
        Boolean or integer array mask

    adata: anndata.AnnData
        The filtered and ordered gene expression

    Returns
    -------
    np.ndarray
    The gene expression prompts
    """
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    return adata[msk, :].X.toarray()