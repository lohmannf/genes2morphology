# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
# Based on code from https://github.com/autonomousvision/stylegan-t
"""Generate images using pretrained network pickle."""

import os
import re
from typing import List, Optional, Union
import PIL.Image

import numpy as np
import torch
import click
import dill
from tqdm import tqdm
import scanpy as sc
from torch.utils.data import DataLoader, TensorDataset

import dnnlib


def parse_range(s: Union[str, List]) -> List[int]:
    '''Parse a comma separated list of numbers or ranges and return a list of ints.

    Example: '1,2,5-10' returns [1, 2, 5, 6, 7]
    '''
    if isinstance(s, list): return s
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        m = range_re.match(p)
        if m:
            ranges.extend(range(int(m.group(1)), int(m.group(2))+1))
        else:
            ranges.append(int(p))
    return ranges


def parse_vec2(s: Union[str, tuple[float, float]]) -> tuple[float, float]:
    '''Parse a floating point 2-vector of syntax 'a,b'.

    Example:
        '0,1' returns (0,1)
    '''
    if isinstance(s, tuple): return s
    parts = s.split(',')
    if len(parts) == 2:
        return (float(parts[0]), float(parts[1]))
    raise ValueError(f'cannot parse 2-vector {s}')


def make_transform(translate: tuple[float,float], angle: float) -> np.ndarray:
    m = np.eye(3)
    s = np.sin(angle/360.0*np.pi*2)
    c = np.cos(angle/360.0*np.pi*2)
    m[0][0] = c
    m[0][1] = s
    m[0][2] = translate[0]
    m[1][0] = -s
    m[1][1] = c
    m[1][2] = translate[1]
    return m

@click.command("cli", context_settings={'show_default': True})
@click.option('--network_pkl', help='Network pickle filename',                  type=str, required=True)
@click.option('--seeds',                  help='List of random seeds (e.g., \'0,1,4-6\')', type=parse_range, required=True)
@click.option('--gex',                    help='Prompt file, .h5ad or .npy',      type=str)
@click.option('--n-samples',              help='Number of prompts sampled (per cluster or prompt)',  type=int)
@click.option('--clusters',               help='File with cluster labels',                 type=str)
@click.option('--outdir',                 help='Where to save the output images',          type=str, required=True)
@click.option('--truncation',             help='Truncation strength',                      type=float, default=1.0)
@click.option('--noise-mode',             help='Noise mode',                               type=click.Choice(['const', 'random', 'none']), default='const')
@click.option('--translate',              help='Translate XY-coordinate (e.g. \'0.3,1\')', type=parse_vec2, default='0,0')
@click.option('--rotate',                 help='Rotation angle in degrees',                type=float, default=0)
@click.option('--device',                 help='CPU or GPU.',                              type=torch.device, default=torch.device('cuda'))
@click.option('--encoded',                help='Whether gene expression is already encoded',is_flag=True)
@click.option('--batch-sz',               help='Batch size for generation',                type=int, default=1)
@click.option('--same-seed',              help='Whether to use the same seed for each prompt',is_flag=True)
def generate_images(
    network_pkl: str,
    seeds: List[int],
    gex: Optional[str],
    n_samples: int,
    clusters: Optional[str],
    outdir: str,
    truncation: float,
    noise_mode: str,
    translate: tuple[float,float],
    rotate: float,
    device: torch.device,
    encoded: bool,
    batch_sz: int,
    same_seed: bool
) -> None:
    print(f'Loading networks from "{network_pkl}"...')
    with dnnlib.util.open_url(network_pkl) as f:
        G = dill.load(f)['G_ema']
    G = G.eval().requires_grad_(False).to(device)
    if G.c_dim > 1:
        assert gex, "Provide a prompt for conditional generators."

    if gex.endswith(".h5ad"):
        adata = sc.read_h5ad(gex)
        sc.pp.normalize_total(adata)
        sc.pp.log1p(adata)

        if clusters:
            idx = []
            clusters = np.load(clusters)
            np.random.seed(1)
            for l in np.unique(clusters):
                idx.append(np.random.choice(np.arange(len(clusters))[clusters == l], size=n_samples, replace=False))
        else:
            idx = [np.random.choice(adata.n_obs, size=n_samples, replace=False)]
    elif gex.endswith(".npy"):
        idx = [np.load(gex)]

    else:
        raise ValueError("Invalid prompt file encountered. Provide .h5ad file or .npy file")

    os.makedirs(outdir, exist_ok=True)

    for seed in tqdm(seeds):
            # Construct an inverse rotation/translation matrix and pass to the generator.  The
            # generator expects this matrix as an inverse to avoid potentially failing numerical
            # operations in the network.
            if hasattr(G.synthesis, 'input'):
                m = make_transform(translate, rotate)
                m = np.linalg.inv(m)
                G.synthesis.input.transform.copy_(torch.from_numpy(m))

            for i, prompts in enumerate(idx):
                rand_state = np.random.RandomState(seed)

                if clusters:
                    os.makedirs(os.path.join(outdir, f"cluster_{i}"), exist_ok=True)
                    prompts = adata[prompts,:].X.toarray()

                loader = DataLoader(TensorDataset(torch.from_numpy(prompts).float()), batch_size=batch_sz, drop_last=False)

                j = 0
                for prompt in tqdm(iter(loader)):
                    prompt = prompt[0].repeat(1 if clusters else n_samples, 1)
                    if same_seed:
                        z = np.random.RandomState(seed).randn(1 if clusters else n_samples, G.z_dim).repeat(prompt.size()[0]//n_samples, 1)
                    else:
                        z = rand_state.randn(prompt.size()[0], G.z_dim)
                    z = torch.from_numpy(z).float().to(device)
                    if encoded:
                        c = torch.zeros((prompt.size()[0] ,18248)).to(device=device).float()
                    else:
                        c = prompt.to(device)

                    w = G.mapping(z, c, truncation_psi=truncation)

                    if encoded:
                        w[:, :, G.z_dim:] = prompt.float().to(device).unsqueeze(1).repeat(1, w.size()[1], 1)

                    img = G.synthesis(w, noise_mode=noise_mode)
                    # Save
                    img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
                    dir = os.path.join(outdir, f"cluster_{i}") if clusters else outdir
                    for image in img:
                        PIL.Image.fromarray(image.cpu().numpy(), 'RGB').save(f'{dir}/prompt_{j:05d}_{f"seed{seed:04d}" if same_seed else ""}.png')
                        j += 1

if __name__ == "__main__":
    generate_images() # pylint: disable=no-value-for-parameter
