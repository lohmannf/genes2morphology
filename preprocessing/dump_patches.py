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

import h5py
import os
from PIL import Image
import sys
import click
import numpy as np
from torchstain.numpy.normalizers import NumpyMacenkoNormalizer
import openslide
import torch

def get_stain_vectors(file: str,
                      scale: float,
                      level: int = 0):
    """
    Get the stain vectors of the center crop of the WSI in file.
    Scale indicates what fraction of the whole image around the center is used to compute the stain vectors

    Parameters
    ----------
    file: str
        Path to WSI

    scale: float
        Relative size of the center crop, must be in (0,1]
    
    level: int
        Pyramidal level at which to extract the center crop.
        Higher level increases computation speed but decreases accuracy

    Returns
    -------
    HE: np.ndarray
        Matrix of H and E stain vectors

    maxC: np.ndarray
        Maximum concentration of the image

    """
    
    wsi = openslide.OpenSlide(file)
    w, h = wsi.level_dimensions[0]

    offset_w = int(w*(1-scale) //2)
    offset_h = int(h*(1-scale) //2)
    w_t = int(w * scale // 2**level)
    h_t = int(h * scale // 2**level)

    center = wsi.read_region((offset_w, offset_h), level = level, size = (w_t, h_t))
    center = center.convert("RGB")
    wsi.close()

    center = np.array(center)
    norm = NumpyMacenkoNormalizer()
    norm.fit(center)
    
    return norm.HERef, norm.maxCRef


def apply_macenko(img: np.ndarray,
                  heMat: np.ndarray,
                  maxC: np.ndarray,
                  heMat_target = [[0.5626, 0.2159],
                                [0.7201, 0.8012],
                                [0.4062, 0.5581]],
                  maxC_target = [1.9705, 1.0308],
                  Io = 240):
    
    '''
    Apply Macenko stain normalization with known stain vectors to a patch.
    Use when the stain vectors and max concentration of the full image are already known.

    Parameters
    ----------
    img: np.ndarray
        Image to normalize

    heMat: np.ndarray
        Matrix of stain vectors for the image

    maxC: np.ndarray
        Maximum concentration of the image

    heMat_target: np.ndarray
        Target stain vectors which the image is normalized to

    maxC_target: np.ndarray
        Target maximum concentration which the image is normalized to
    
    Io: int
        Maximum pixel value

    Returns
    -------
    Inorm: np.ndarray
        Stain-normalized image
    '''
    img_shape = img.shape #channel-last

    if isinstance(heMat_target, list):
        heMat_target = np.array(heMat_target)
        maxC_target = np.array(maxC_target)
        
    img = img.reshape((-1, 3))

    OD = -np.log((img.astype(float)+1)/Io)
    Y = np.reshape(OD, (-1, 3)).T
    C = np.linalg.lstsq(heMat, Y, rcond=None)[0]

    maxC = np.divide(maxC, maxC_target)
    C = np.divide(C, maxC[:, np.newaxis])

    Inorm = np.multiply(Io, np.exp(-heMat_target.dot(C)))
    Inorm[Inorm > 255] = 255
    Inorm = np.reshape(Inorm.T, img_shape).astype(np.uint8)

    return Inorm

def parse_list(s):
    """Parse comma separated list into list of strings"""
    if isinstance(s, list):
        return s
    return s.split(",")

def find_wsi_file(id: str, dir: str):
    try:
        return [f for f in os.listdir(dir) if f.strip(".tif").endswith(id)][0]
    except IndexError:
        raise ValueError(f"No matching wsi with id {id} found") 

def pad_label(x, max_lab):
    x = str(x)
    max_lab = str(max_lab)
    nzeros = len(max_lab)-len(x)
    return "0"*nzeros + x

def dump_patches_to_folder_single(images_path, wsi_path, out_dir, normalize: bool = True):
    """
    Write image patches from a hdf file to a folder and optionally apply stain normalization

    Parameters
    ----------
    images_path: str
        Path to the HDF file containing the images

    wsi_path: str
        Path to the tif file containing the WSI for stain vector extraction

    out_dir: str
        Output directory that patches are written to as individual .png files

    normalize: bool
        Whether to apply stain normalization, default = True

    Returns
    -------
    None
    """
    os.makedirs(out_dir, exist_ok=True)
    if normalize:
        HEr, Cr = get_stain_vectors(wsi_path, level = 1, scale = 0.8)
    with h5py.File(images_path) as file:
        lmax = file["img"].shape[0]-1
        for i, img in enumerate(file["img"]):
            if normalize:
                img = apply_macenko(img, HEr, Cr)
            Image.fromarray(img).save(os.path.join(out_dir, f"patch_{pad_label(i, lmax)}.png"), format='png', compress_level=0, optimize=False)

@click.option("--ids",            help="List of sample ids, e.g. A,B,C",                                        type=str, required=True)
@click.option("--root-dir",       help="Contains subdirectory with hdf file with image patches for each sample",type=str, default=".")
@click.option("--out-dir",        help="Root directory for outputs, will create subdirectory for each sample",  type=str, default=".")
@click.option("--normalize",      help="Whether to perform stain normalization",                                default=True, is_flag=True)
@click.option("--wsi-dir",        help="Directory containing WSIs for each sample, WSI files must end with id", type=str, default=".")
def dump_patches_to_folder(ids: list, root_dir: str, out_dir: str, normalize, wsi_dir: str):
    """
    Write image patches from a HDF file to a folder with structured names and optionally apply stain normalization
    """
    wsi_paths = [os.path.join(wsi_dir, find_wsi_file(id, wsi_dir)) for id in ids]
    paths = [os.path.join(root_dir, id, "imgs.h5") for id in ids]

    for id, p, wsi_p in zip(ids, paths, wsi_paths):
        dump_patches_to_folder_single(p, wsi_p, os.path.join(out_dir, id), normalize)

if __name__ == "__main__":
    dump_patches_to_folder() #pylint: disable=no-value-for-parameter