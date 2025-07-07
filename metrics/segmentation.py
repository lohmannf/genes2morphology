import torch.nn as nn
import torch.nn.functional as F
import torch
import sys
import torchvision.transforms as tvt
from typing import Dict
import numpy as np

sys.path.append("/home/lohmanf1/") # hack to make the import work
sys.path.append("/home/lohmanf1/cellvit/")
from cellvit.models.segmentation.cell_segmentation.cellvit import CellViTSAM
from cellvit.utils.tools import unflatten_dict
from cellvit.cell_segmentation.utils.post_proc_cellvit import DetectionCellPostProcessor

def calculate_instance_map(
        predictions: Dict[str, torch.Tensor], num_nuclei_classes: int, magnification: int = 40
    ):
        # reshape to B, H, W, C
        predictions_ = predictions.copy()
        predictions_["nuclei_type_map"] = predictions_["nuclei_type_map"].permute(
            0, 2, 3, 1
        )
        predictions_["nuclei_binary_map"] = predictions_["nuclei_binary_map"].permute(
            0, 2, 3, 1
        )
        predictions_["hv_map"] = predictions_["hv_map"].permute(0, 2, 3, 1)

        cell_post_processor = DetectionCellPostProcessor(
            nr_types=num_nuclei_classes, magnification=magnification, gt=False
        )
        instance_preds = []
        type_preds = []

        for i in range(predictions_["nuclei_binary_map"].shape[0]):
            pred_map = np.concatenate(
                [
                    torch.argmax(predictions_["nuclei_type_map"], dim=-1)[i]
                    .detach()
                    .cpu()[..., None],
                    torch.argmax(predictions_["nuclei_binary_map"], dim=-1)[i]
                    .detach()
                    .cpu()[..., None],
                    predictions_["hv_map"][i].detach().cpu(),
                ],
                axis=-1,
            )
            instance_pred = cell_post_processor.post_process_cell_segmentation(pred_map)
            instance_preds.append(instance_pred[0])
            type_preds.append(instance_pred[1])

        return torch.Tensor(np.stack(instance_preds)), type_preds

def cell_count(pred, bkg_label, n_only: bool = True, magnification: int = 20):
        pred["nuclei_binary_map"] = F.softmax(
            pred["nuclei_binary_map"], dim=1
        )  # shape: (batch_size, 2, H, W)
        pred["nuclei_type_map"] = F.softmax(
            pred["nuclei_type_map"], dim=1
        )  # shape: (batch_size, num_nuclei_classes, H, W)
        _, instance_types = calculate_instance_map(pred, num_nuclei_classes=pred["nuclei_type_map"].shape[1], magnification=magnification)
        if n_only:
            res = []
        for patch_instance_types in instance_types:
            cell_count = 0
            for cell in patch_instance_types.values(): # centroid, contour, bbox
                if cell["type"] == bkg_label:
                    continue
                cell_count += 1
            if n_only:
                res.append(cell_count)
            else:
                patch_instance_types["n_cells"] = cell_count
        if n_only:
            res = torch.tensor(res)#.to(instance_types)

        return res if n_only else instance_types

class CellViT(nn.Module):
    """
    Wrapper class for CellViT model.
    Ported from https://github.com/TIO-IKIM/CellViT/blob/main/cell_segmentation/inference/cell_detection.py
    """

    def __init__(self, model_path: str):
        super().__init__()

        ckpt = torch.load(model_path, map_location = "cpu")
        
        self.conf = unflatten_dict(ckpt["config"])
        self.vit = CellViTSAM(model_path = None,
                              num_nuclei_classes = self.conf["data"]["num_nuclei_classes"],
                              num_tissue_classes=self.conf["data"]["num_tissue_classes"],
                              vit_structure=self.conf["model"]["backbone"],
                              regression_loss=self.conf["model"].get("regression_loss", False)
                              )
        self.vit.load_state_dict(ckpt["model_state_dict"])

        self.mean = self.conf["transformations"]["normalize"].get("mean", (0.5, 0.5, 0.5))
        self.std = self.conf["transformations"]["normalize"].get("std", (0.5, 0.5, 0.5))
        self.bkg_label = self.conf["dataset_config"]["nuclei_types"]["Background"]

    def forward(self, x: torch.Tensor, resize: bool = False, div255: bool = False):
        if div255:
            x = x.to(torch.float32) / 255.
        if resize:
            x = nn.functional.interpolate(x, size=(256, 256), mode = "area")
        x = tvt.functional.normalize(x, mean=self.mean, std=self.std)
        return self.vit(x, retrieve_tokens = False)

if __name__ == "__main__":
    import os
    
    root = "../models/"
    ckpt = os.path.join(root, "CellViT-SAM-H-x40.pth")

    model = CellViT(ckpt)
    model = torch.jit.script(model)
    model.save(os.path.join(root, "cellvit40X.ts"))

