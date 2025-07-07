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
import torch.nn as nn
from captum.attr import IntegratedGradients, NoiseTunnel, Saliency
import click
import numpy as np
from networks import clip
import scanpy as sc
import pandas as pd

class SumOfSquares(nn.Module):
    """
    Squared L2 distance
    """
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x1, x2, **model_kwargs):
        x1 = self.model.encode_gex(x1, **model_kwargs)
        x2 = self.model.encode_gex(x2, **model_kwargs)
        return (x1-x2).pow(2).sum(dim=1)
    

def get_attributions(model, x1, x2, only_upregulated: bool = True, method: str = "ig"):
    """
    Get attribution scores for squared L2 distance between query (x1) and reference (x2) inputs
    in latent space induced by model

    Parameters
    ----------
    model: torch.nn.Module
        A model that outputs an embedding of the input

    x1: torch.Tensor
        Query input

    x2: torch.Tensor
        Reference input

    only_upregulated: bool
        Assign score of 0 to all input activations that are smaller in the query than in the reference

    method: str
        Attribution method to use, "ig" for Integrated Gradients, "sg" for SmoothGrad

    Returns
    --------
    attrs: torch.Tensor
        Attribution scores for each input activation
    """
    
    model.zero_grad()

    if method == "ig":
        explainer = IntegratedGradients(SumOfSquares(model))
        attrs = explainer.attribute(x1, baselines=x2, additional_forward_args = x2)
    elif method == "sg":
        explainer = NoiseTunnel(Saliency(SumOfSquares(model)))
        stdevs = tuple(((x1.max(dim=0)-x1.min(dim=0))*0.15).tolist())
        attrs = explainer.attribute(x1, additional_forward_args = x2, nt_type="smoothgrad", stdevs=stdevs, nt_samples=50)
    else:
        raise ValueError(f"Unknown method {method}")

    if only_upregulated:
        attrs *= x1 > x2

    attrs = +attrs.abs()

    if not only_upregulated:
        attrs[x1 < x2] *= -1

    return attrs

@click.command('cli', context_settings={'show_default': True})
@click.option('--network',      help='Network weights filename',    type=str, required=True)
@click.option('--gex',          help='Gene expression filename',    type=str, required=True)
@click.option('--cluster-labs', help='Cluster labels filename',     type=str, required=True)
@click.option('--method',       help='Attribution method',          type=click.Choice(["ig", "sg"]), default= "ig")
@click.option('--out-file',     help='CSV output filename',         type=str, required=True)
@click.option('--device',       help='Torch device',                type=torch.device, default='cuda')
def main(network: str,
         gex: str,
         cluster_labs: str,
         method: str,
         out_file: str,
         device: torch.device
         ):
    '''
    Determine attribution scores for all genes between average gene expression of pairs of clusters.
    Computes scores for every unique query-reference cluster combination.

    Expects cluster labels for spots to be provided as a numpy file in same order as the gene expression file.
    Expects gene expression to be preprocessed (genes filtered and ordered)
    '''
    assert out_file.endswith(".csv"), "Provide CSV filename for outputs"
    
    # Load gene expression and cluster labels
    clusters = np.load(cluster_labs)
    gex = sc.read_h5ad(gex)
    sc.pp.normalize_total(gex)
    sc.pp.log1p(gex)

    gex_cluster = []

    for i in np.unique(clusters):
        gex_cluster.append(torch.from_numpy(gex[clusters == i, :].X.mean(axis=0).flatten()).to(torch.float32))

    gex1 = []
    gex2 = []
    for i, g in enumerate(gex_cluster):
        gex1.append(g.repeat(len(gex_cluster)-1,1))
        gex2.extend([x for x in gex_cluster if (x != g).any()])

    gex1 = torch.cat(gex1, dim=0).to(device)
    gex2 = torch.cat(gex2, dim=0).to(device)

    model = clip.CLIP(network).to(device).eval()

    att = get_attributions(model, gex1, gex2, only_upregulated=True, method=method)
    cols = [f"{i//(len(gex_cluster)-1)} vs. {i%(len(gex_cluster)-1) + (i%(len(gex_cluster)-1) >= i//(len(gex_cluster)-1))}" for i in range(len(att))]
    pd.DataFrame(data = att.cpu().T.numpy(), index = gex.var_names.tolist(), columns = cols).to_csv(out_file)


if __name__ == "__main__":
    main() #pylint: disable=no-value-for-parameter