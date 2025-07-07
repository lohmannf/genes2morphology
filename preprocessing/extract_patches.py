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

import geopandas as gp
from shapely.geometry import LinearRing
import numpy as np
import h5py
from tqdm import tqdm
import torch
import scanpy as sc
import openslide
from torchvision.transforms.functional import pil_to_tensor, resize
import anndata as ad
import pandas as pd
import json
import os
from scipy import sparse
import click

def write_patches(coords, n_pts: int, extract_sz: tuple, target_sz: tuple, tif_file: str, out_file:str, bcs: list = None, thresh_img_mean: float = 256, update_freq: int = 1):
    n_faulty = 0
    n_written = 0
    wt_og, ht_og = target_sz
    images = [] 

    if bcs is not None:
        valid_bcs = []
        
    wsi = openslide.OpenSlide(tif_file)

    prog_bar = tqdm(desc = "Extracting patches", total=n_pts)
    for i, coords in enumerate(coords):
        # extract image at the lowest level and resize to the proper resolution
        samp = wsi.read_region(location=coords, level=0, size=extract_sz).convert(mode = "RGB")
        samp = pil_to_tensor(samp)
        samp = resize(samp, (ht_og, wt_og))

        if thresh_img_mean < 256. and samp.float().mean() > thresh_img_mean:    
            n_faulty += 1
        else:
            images.append(samp.permute(1,2,0)) #torch.cat([images, samp[None, :,:,:]], axis = 0)
            n_written += 1
            if n_written % update_freq == 0:
                prog_bar.update(update_freq)

            if bcs is not None:
                valid_bcs.append(bcs[i])

        if n_written >= n_pts:
            break

    print(f"Removed {n_faulty} images outside of tissue")

    with h5py.File(out_file, "w") as f:
        f["img"] = torch.stack(images)
        if bcs is not None:
            f["barcode"] = np.array([b.encode("utf-8") for b in valid_bcs]).astype(object)

    prog_bar.close()
    print(f"Wrote {n_written}/{n_pts} images in total")
    wsi.close()
    if bcs is not None:
        del valid_bcs
    del images


def aggregate_gex(bcs, bins, adata: ad.AnnData, update_freq: int = 1):
    agg_gex = sparse.lil_matrix((len(bcs), adata.n_vars))
    
    for i, patch in enumerate(tqdm(bins, desc="Aggregating gene expression", miniters=update_freq)):
        agg_gex[i] = adata[patch].X.sum(0)

    agg_gex = agg_gex.tocsr()

    adata_agg = ad.AnnData(X = agg_gex, obs = pd.DataFrame(index=bcs))
    adata_agg.var_names = adata.var_names
    return adata_agg

def patchify_10XHD(bin_file: str,
                pos_file: str,
                meta_json: str,
                tif_file: str,
                out_file_adata: str,
                out_file_patch: str,
                target_size: tuple[int, int],
                patch_sz_um: float = 55.,
                stride: int = 1,
                dump_coords: bool = False,
                update_freq: int = 10000
                ):
    
    """
    Extract all valid unique image patches of the specified size and paired gene expression from a Visium HD sample.

    Unique patches are acquired with stride 2um*stride, valid refers to whether all 2 um bins covered by the patch are available.

    Parameters
    -----------
    bin_file: str
        Path to filtered feature matrix in hdf5 format

    pos_file: str
        Path to tissue positions file. Can be csv or parquet.

    meta_json: str
        Path to spatial metadata file of 2um bin configuration

    tif_file: str
        Path to file with full resolution brightfield H&E image. Must be compatible with OpenSlide

    out_file_adata: str
        Location where gene expression per patch is written in anndata format

    out_file_patch: str
        Location where image patches are written in hdf5 format

    target_size: tuple[int, int]
        Final size of the image patches. Patches will be resized regardless
        of their original size at extraction.

    patch_sz_um: float
        Absolute size of the region covered by a patch in um, default = 55.
        Dictates the patch size at extraction.

    stride: int
        Stride in number of 2um bins used when extracting adjacent patches, default = 1

    update_freq: int
        Update frequency of the progress bar to avoid cluttering of output files, default = 10000

    dump_patches: bool
        If ```True```, only write the top left coordinates and barcodes of member bins to ```out_file_patch```.
        Default = False
    
    Returns
    --------
    None
    """
    
    with open(meta_json, "r") as f:
        facs = json.load(f)
    bins_per_patch = int(patch_sz_um / facs["bin_size_um"])
    assert bins_per_patch % 2 == 1

    patch_sz_px = int(patch_sz_um / facs["microns_per_pixel"])

    gex = sc.read_10x_h5(bin_file)
    gex.var_names_make_unique()

    if pos_file.endswith(".parquet"):
        pos = pd.read_parquet(pos_file)
    elif pos_file.endswith(".csv"):
        pos = pd.read_csv(pos_file)
    else:
        raise NotImplementedError("Unknown file type for tissue positions")
    
    # filter out background bins
    pos = pos[pos.in_tissue.astype(bool)]
    pos = pos.set_index("barcode")
    assert len(pos) == gex.n_obs, "Number of barcodes in position and gene expression file does not match"

    gex.obs = pd.merge(gex.obs, pos, left_index = True, right_index = True)
    del pos

    gex_meta = gex.obs.sort_values(by=["array_row", "array_col"], ascending=[False, True])

    rmin = gex_meta.array_row.iloc[-1]
    rmax = gex_meta.array_row.iloc[0]
    cmin = gex_meta.array_col.min()
    cmax = gex_meta.array_col.max()

    # valid bins in row-column fashion
    exists = np.zeros((rmax-rmin+1, cmax-cmin+1), dtype = bool)
    exists[gex_meta.array_row.values - rmin, gex_meta.array_col.values - cmin] = True

    # patch barcodes accessible in row-column fashion
    bcs = np.zeros((rmax-rmin+1, cmax-cmin+1), dtype = object)
    bcs[gex_meta.array_row.values - rmin, gex_meta.array_col.values - cmin] = gex_meta.index.to_numpy()


    #idx = start_bin
    patch_bc = []
    tl_xy = []
    bins = []

    for ridx in tqdm(range(exists.shape[0]-1, (bins_per_patch-1)-1, -stride), unit_scale=exists.shape[1], desc="Identifying valid patches"):
        for cidx in range(0, exists.shape[1]-(bins_per_patch-1), stride):
            # get all bins that are part of the corresponding patch
            rrange = (-1 * np.arange(bins_per_patch) + ridx - rmin)[:, np.newaxis].astype(int)
            crange = (np.arange(bins_per_patch) + cidx - cmin).astype(int)

            if (~exists[rrange, crange]).sum() > 0:
                # at least 1 missing bin, cannot extract the patch
                continue

            # Extract centroid coordinates
            cent_coords = gex_meta.loc[bcs[int(ridx - bins_per_patch//2), int(cidx + bins_per_patch//2)], ["pxl_col_in_fullres", "pxl_row_in_fullres"]].values
            cent_coords = (cent_coords - patch_sz_px/2).astype(int)
            if not dump_coords:
                cent_coords = tuple(cent_coords)
            curr_bcs = bcs[rrange, crange].flatten()

            patch_bc.append(curr_bcs[bins_per_patch**2//2])
            tl_xy.append(cent_coords)
            bins.append(curr_bcs)

    print(f"Identified {len(patch_bc)} valid patches", flush=True)

    # conserve memory
    del exists
    del bcs
    del gex_meta

    if dump_coords:
        with h5py.File(out_file_patch, "w") as file:
            file["xy"] = np.array(tl_xy)
            file["barcode"] = np.array([b.encode("utf-8") for b in patch_bc]).astype(object)
            file["bins"] = np.array([[b.encode("utf-8") for b in x] for x in bins])
            
    else:
        write_patches(tl_xy, len(patch_bc), (patch_sz_px, patch_sz_px), target_size, 
                    tif_file, out_file_patch, patch_bc, update_freq = max(1, update_freq))
        del tl_xy

        gex_agg = aggregate_gex(patch_bc, bins, gex, max(1, update_freq))
        gex_agg.write_h5ad(out_file_adata)
        del gex_agg
        del gex


def patchify(file: str, 
             tif_file: str, 
             out_file: str, 
             level: int, 
             px_sz_true: float,
             pos_file: str = None,
             n_pts: int = 10000, 
             n_buffer: int = 2000, 
             target_size: tuple[int, int] = (256, 256), 
             thresh_img_mean: float = 255., 
             px_sz_ref_0: float = 0.25,
             spacer_old: bool = False):
    
    """
    Extract patches from a whole slide image

    Parameters
    ----------
    file: str
        Path to either a geojson file in the style of the HEST library holding the tissue boundaries for sampling random patches 
        or h5ad file / 10x h5 file containing the gene expression of individual spots for extracting spot-centered patches

    tif_file: str
        Path to pyramidal file that contains the whole slide image

    out_file: str
        Path to the hdf5 file that the patches are written to 
    
    level: int
        Level at which to extract the patches, relative to highest resolution level

    px_sz_true: float
        Actual size of a pixel in the tiff file at level 0 in um

    pos_file: str
        Path to the csv file containing space ranger tissue positions. Only used if ```file``` is a 10X hdf5 file.
        Default = None

    n_pts: int
        Number of patches to sample. Will only be used if file is a geojson, default = 10000

    n_buffer: int
        Number of backup patches that can replace patches which are above ```thresh_img_mean```, default = 2000

    target_size: tuple[int, int]
        Size of the final patches in pixels, default = (256, 256)

    thresh_img_mean: float
        Simple threshold on the mean of an image that is used to filter background patches
        at extraction time, default = 255.

    px_sz_ref_0: float
        Desired reference size of a pixel at highest resolution. The pixel size in the final extracted patches will be ```px_sz_ref_0*2**level```.
        Bilinear interpolation will be used to coerce ```px_sz_true``` to ```px_size_ref_0```, will lead to uninformative
        pixels if ```px_sz_true > px_size_ref_0```, default = 0.25

    spacer_old: bool
        Indicates whether the tissue file was created with Space Ranger older than v2.0.
        If true, assumes that ```pos_file``` does not contain headers, default = False.
        
    Returns
    --------
    None
    """

    w_t, h_t = target_size
    sample = file.endswith(".geojson")
    scale_factor = px_sz_ref_0/px_sz_true

    ht_og, wt_og = h_t, w_t
    h_t = int(h_t*scale_factor)
    w_t = int(w_t*scale_factor)

    w, h =  (w_t* 2**level, h_t * 2**level)

    print(f"Extracting patches of size {h}x{w} on level 0 to reach the desired resolution of {px_sz_ref_0 * 2**level} um/px on level {level}")

    valid_pts = []

    if sample:
        tissue = gp.read_file(file)

        discarded = 0
        while len(valid_pts) < n_pts + n_buffer:

            x, y = tissue.sample_points(1).iloc[0].coords[0]
            x = int(x)
            y = int(y)
            patch = LinearRing(coordinates = (
                                (x, y), 
                                (x, y+h),
                                (x+h, y+h), 
                                (x+h, y)))

            if tissue.geometry.contains(patch).any():
                valid_pts.append((x, y))
            else:
                discarded += 1

        print(f"Resampled {discarded} patches because they exceeded tissue bounds")

    else:
        if file.endswith("filtered_feature_bc_matrix.h5"):
            assert pos_file is not None
            adata = sc.read_10x_h5(file)
            pos = pd.read_csv(pos_file, header = None if spacer_old else 0, index_col=0)
            if spacer_old:
                pos = pos.rename(columns = {1: "in_tissue", 2: "array_row", 3: "array_col", 4:"pxl_row_in_fullres", 5:"pxl_col_in_fullres"})
            adata.obs = pd.merge(adata.obs, pos, left_index=True, right_index=True)
            centers = zip(adata.obs.pxl_col_in_fullres.values, adata.obs.pxl_row_in_fullres.values)
            del pos

        else:
            adata = sc.read_h5ad(file)
            if pos_file is not None:
                with h5py.File(pos_file) as f:
                    centers = f["xy"][:]
            else:
                centers = adata.obsm["spatial"]

        for cent in centers:
            valid_pts.append((int(cent[0]-w/2), int(cent[1]-h/2)))

        n_pts = len(valid_pts)
        bcs = np.array(list(adata.obs_names))
        del adata
        del centers

    write_patches(valid_pts, n_pts, (w, h), (wt_og, ht_og), tif_file, out_file, None if sample else bcs, thresh_img_mean)

@click.command("cli", context_settings={'show_default': True})
@click.option("--space-ranger-dir", type=str, default=None, help="Directory with Space Ranger outputs.")
@click.option("--wsi-file", type=str, help="Path to full resolution pyramidal tiff image")
@click.option("--is-hd", is_flag=True, help="Whether the sample is Visium HD or not")
@click.option("--tissue-seg-file", type=str, default=None, help="Path to geojson file with tissue contours. Use instead of space-ranger-dir for random sampling")
@click.option("--out-file", type=str, default=None, help="Path to output image file. Only used when sampling random patches")
@click.option("--n-pts", type=int, default=50000, help="Number of patches to extract. Ignored if tissue-seg-file is not defined")
@click.option("--stride", type=int, default=5 ,help="Stride in # bins for pseudo-spot acquistion in VisiumHD")
@click.option("--patch-sz-px", type=int, default=128, help="Target size of output patches in pixels")
@click.option("--patch-sz-um", type=float, default=55, help="Target size of output patches in microns")
@click.option("--use-wsi-res", is_flag=True, help="Whether to use internally stored WSI resolution instead of scalefactors_json information. Ignored for VisiumHD")
@click.option("--below-v2", is_flag=True, help="Whether the Space Ranger version used for processing is below V2")
def main(space_ranger_dir, wsi_file, is_hd, tissue_seg_file, out_file, n_pts, stride, patch_sz_px, patch_sz_um, use_wsi_res, below_v2):
    if space_ranger_dir is not None and tissue_seg_file is not None:
        raise ValueError("Cannot use tissue contours and spot positions simultaneously")
    
    if tissue_seg_file is not None:
        patchify(file = tissue_seg_file,
             tif_file = wsi_file,
             out_file = out_file,
             level = 0,
             px_sz_true=float(openslide.OpenSlide(wsi_file).properties["openslide.mpp-x"]),
             n_pts = n_pts,
             n_buffer = 0,
             target_size=(patch_sz_px, patch_sz_px),
             px_sz_ref_0=patch_sz_um/patch_sz_px,
            )

    elif is_hd:
        patchify_10XHD(bin_file = os.path.join(space_ranger_dir, "filtered_feature_bc_matrix.h5"),
                        pos_file = os.path.join(space_ranger_dir, "spatial/tissue_positions.parquet"),
                        meta_json = os.path.join(space_ranger_dir, "spatial/scalefactors_json.json"),
                        tif_file = wsi_file,
                        out_file_adata= os.path.join(space_ranger_dir, "gex.h5ad"),
                        out_file_patch= os.path.join(space_ranger_dir, "imgs.h5"),
                        target_size=(patch_sz_px, patch_sz_px),
                        patch_sz_um=patch_sz_um,
                        stride = stride,
                        update_freq = 5000,
                        dump_coords = False,
                   )
        
    else:
            if use_wsi_res:
                px_sz_true = float(openslide.OpenSlide(wsi_file).properties["openslide.mpp-x"])
            else:
                with open(os.path.join(space_ranger_dir, "spatial/scalefactors_json.json"), "r") as file:
                    px_sz_true = patch_sz_um/json.load(file)["spot_diameter_fullres"]

            patchify(file = os.path.join(space_ranger_dir, "filtered_feature_bc_matrix.h5"),
                    tif_file = wsi_file,
                    out_file = os.path.join(space_ranger_dir, "imgs.h5"),
                    level=0,
                    px_sz_true=px_sz_true, 
                    pos_file=os.path.join(space_ranger_dir, "spatial/tissue_positions.csv"),
                    target_size = (patch_sz_px, patch_sz_px),
                    px_sz_ref_0 = patch_sz_um/patch_sz_px,
                    spacer_old=below_v2)


if __name__ == "__main__":
    main()