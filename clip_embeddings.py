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

import torch
from networks.clip import CLIP
import dnnlib
import math
from tqdm import tqdm
import numpy as np
import os
import click

@click.command('cli', context_settings={'show_default': True})
@click.option('--network', 'network_path', help='Network weights filename',                  type=str, required=True)
@click.option('--gex',                     help='Path to gex h5ad file',                     type=str, required=True)
@click.option('--images',                  help='Path to image folder',                      type=str, required=True)
@click.option('--out-dir',                 help='Output directory',                          type=str, default=".")
@click.option('--batch-size',              help='Inference batch size',                      type=int, default=64)
def main(network_path: str, gex: str, images: str, out_dir: str, batch_size: int):
    """
    Get Image and Gene Expression CLIP embeddings of spatial transcriptomics sample.

    Expects preprocessed gene expression (genes filtered and sorted in correct order) and an image folder
    with file names in the same order as occurrence of (pseudo-) spots in the gene expression.
    """
    torch.manual_seed(1)
    data_loader_kwargs = dnnlib.EasyDict(pin_memory=True, num_workers=2, prefetch_factor=2)
    training_set_kwargs = dnnlib.EasyDict(path=images, resolution=128, gex_path=gex,
        class_name="training.data_zip.ImageGeneDataset", xflip=False, use_labels=False, max_size=None)

    model = CLIP(network_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    training_set = dnnlib.util.construct_class_by_name(**training_set_kwargs)
    training_set_iterator = iter(torch.utils.data.DataLoader(dataset=training_set, batch_size=batch_size, **data_loader_kwargs))

    img_embs = []
    gex_embs = []
    with torch.no_grad():
        for img, gex in tqdm(training_set_iterator, total=math.ceil(len(training_set)/batch_size)):
            img = img.to(device).to(torch.float32)
            gex = gex.to(device).to(torch.float32)

            img_embs.append(model.encode_image(img, div255=True).cpu())
            gex_embs.append(model.encode_gex(gex).cpu())

    img_embs = torch.cat(img_embs, 0)
    gex_embs = torch.cat(gex_embs, 0)

    np.save(os.path.join(out_dir, "image_clip_emb.npy"), img_embs.numpy(), allow_pickle=False)
    np.save(os.path.join(out_dir, "gene_clip_emb.npy"), gex_embs.numpy(), allow_pickle=False)


if __name__ == "__main__":
    main() #pylint: disable=no-value-for-parameter