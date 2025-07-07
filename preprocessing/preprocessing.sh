#/bin/bash

# List of genes in correct order
VOCAB="../assets/gene_vocab.txt"
# Path to directory with space ranger outputs
ROOT_DIR="../data/space_ranger_outputs"
GEX_DIR="${ROOT_DIR}/TENX45"
IMG_DIR="../data/patches_55um_paired/TENX45"
WSI="..data/wsis"
ID=45

echo "-- Preprocessing data for sample ${ID} with gene data from ${GEX_DIR} and image ${WSI}"

# Extract image patches and gene expression
python3 extract_patches.py --space-ranger-dir $GEX_DIR --wsi-file "${WSI}/TENX45.tif" --below-v2 --use-wsi-res
echo "-- Extracted patches"

# Filter gene expression
python3 gex_preprocessing.py --vocab-file $VOCAB --raw-file "${GEX_DIR}/gex.h5ad"
echo "-- Filtered genes"

# Dump image patches to a folder
python3 dump_patches.py --ids $ID --root-dir $ROOT_DIR --out-dir $IMG_DIR --wsi-dir $WSI --normalize
echo "-- Dumped patches into ${IMG_DIR}"