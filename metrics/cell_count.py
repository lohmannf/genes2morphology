"""
Segment cells from ground truth and generated image and compute the difference between the cell count.
Implementation uses CellViT (Hörst et al. 2023) for image segmentation.
"""
import torch
import numpy as np
import random
from tqdm import tqdm
import os
import copy

import dnnlib
from metrics.segmentation import CellViT, cell_count
from metrics import metric_utils

def compute_cell_count_diff(opts):
    path = "../models/cellvit.ts"
    batch_size = 64
    batch_gen = min(batch_size, 4)
    data_loader_kwargs = dict(pin_memory=True, num_workers=2, prefetch_factor=2)
    if path.endswith("pt"):
        model = CellViT(path)

    elif path.endswith("ts"):
        model = torch.jit.load(path)
    model = model.to(opts.device).eval()

    # get truncation values
    truncation_psi = opts.G_kwargs.get('truncation_psi', 1.0)

    # Setup generator and labels.
    G = copy.deepcopy(opts.G).eval().requires_grad_(False).to(opts.device)

    # Initialize.
    dataset = dnnlib.util.construct_class_by_name(**opts.dataset_kwargs)
    num_items = len(dataset)
    if opts.max_items is not None:
        num_items = min(num_items, opts.max_items)
    stats = metric_utils.FeatureStats(max_items = num_items, capture_mean = True)
    progress = opts.progress.sub(tag='dataset features', num_items=num_items, rel_lo=0, rel_hi=1)

    # Main loop.
    item_subset = [(i * opts.num_gpus + opts.rank) % num_items for i in range((num_items - 1) // opts.num_gpus + 1)]
    random.shuffle(item_subset)

    loader = iter(torch.utils.data.DataLoader(dataset=dataset, sampler=item_subset, batch_size=batch_size, **data_loader_kwargs))

    while not stats.is_full():
        images, c = next(loader)
        G.mapping.num_broadcast = G.mapping.num_ws
        with torch.no_grad():
            z = torch.randn((images.shape[0], G.z_dim), device=opts.device)
            c = c.to(opts.device).to(torch.float32)
            w = G.mapping(z, c, truncation_psi=truncation_psi)
            images_gen = G.synthesis(w)
            images_gen = torch.clamp((images_gen + 1)/2., 0, 1)
            cts_real = cell_count(model(images.to(opts.device), div255=True), bkg_label=model.bkg_label, magnification=40)
            cts_gen = cell_count(model(images_gen.to(opts.device)), bkg_label=model.bkg_label, magnification=40)
            mse = torch.nn.functional.mse_loss(cts_gen.float(), cts_real.float(), reduction='none')[:,None]
        
        stats.append_torch(mse, num_gpus=opts.num_gpus, rank=opts.rank)
        progress.update(stats.num_items)

    del model
    del G
    return float(stats.get_mean())
