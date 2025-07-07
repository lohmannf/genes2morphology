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
import click
import scanpy as sc

@click.command("cli", )
@click.option("--raw-file", type=str, help="Path to the file with raw gene expression data")
@click.option("--vocab-file", type=str, help="Path to txt file with gene vocabulary", default="../assets/gene_vocab.txt")
def filter_genes(raw_file: str, vocab_file: str):
    out_file = os.path.join(os.path.dirname(raw_file), os.path.basename(raw_file).strip(".h5ad") + "_filt.h5ad")
    
    # filter and reorder genes
    adata = sc.read_h5ad(raw_file)
    adata.var_names_make_unique()
    with open(vocab_file) as f:
        vocab = f.readlines()
    adata = adata[:, vocab]

    adata.write_h5ad(out_file)

if __name__ == "__main__":
    filter_genes()
