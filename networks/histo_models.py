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

import timm
from torch import nn
from torchvision import transforms as tvt
import torch

UNI_STATE_DICT_PATH = "../models/uni.pt"
GIGAPATH_STATE_DICT_PATH = "../models/gigapath.pt"

_model_dict = dict()

def register_model(cls=None, *, name=None):

    def _register(cls):
        if name is None:
            local_name = cls.__name__
        else:
            local_name = name
        if local_name in _model_dict:
            raise ValueError(
                f'Already registered model with name: {local_name}')
        _model_dict[local_name] = cls
        return cls

    if cls is None:
        return _register
    else:
        return _register(cls)

def get_model_by_name(name: str, **kwargs):
    return _model_dict[name](**kwargs)

def get_valid_models():
    return list(_model_dict.keys())

@register_model(name = "uni")
class UNIFeatureExtractor(nn.Module):
    def __init__(self, local_path: str = None):
        super().__init__()
        if local_path is None:
            # does not work with old versions of timm
            self.model = timm.create_model("hf-hub:MahmoodLab/uni", pretrained=True, init_values=1e-5, dynamic_img_size=True)
        else:
            self.model = timm.create_model("vit_large_patch16_224", init_values=1e-5, num_classes=0)
            sdict = torch.load(local_path, map_location="cpu")
            self.model.load_state_dict(sdict, strict=True)

        for p in self.model.parameters():
            p.requires_grad = False
        self.eval()

    def forward(self, x, transform: bool=True, div255: bool = False):
        if div255:
            x = x.to(torch.float32) / 255.
        if transform:
            x = nn.functional.interpolate(x, size=(224, 224), mode = "area")
            x = tvt.functional.normalize(x, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        return self.model(x)
    
@register_model(name = "gigapath")
class GigaPathFeatureExtractor(nn.Module):
    def __init__(self, local_path: str = None):
        super().__init__()
        if local_path is None:
            # does not work with old versions of timm
            self.model = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True)
        else:
            # does not work with old versions of timm
            self.model = timm.create_model("vit_giant_patch14_dinov2", num_classes=0, init_values=1e-5, 
                                           pretrained_cfg_overlay={"input_size": [3, 224, 224]}, patch_size=16)
            sdict = torch.load(local_path, map_location="cpu")
            self.model.load_state_dict(sdict, strict=True)

    def forward(self, x, transform: bool=True, div255: bool = False):
        if div255:
            x = x.to(torch.float32) / 255.
        if transform:
            x = nn.functional.interpolate(x, size=(224, 224), mode = "area")
            x = tvt.functional.normalize(x, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        return self.model(x)


def dump_model_to_disk(name: str, out_location: str, as_torchscript: bool = False, **model_kwargs):
    """
    Write a named histopathology model to disk to allow version-agnostic loading.

    Parameters
    ----------
    name: str
        The identifier of the histopathology model

    out_location: str
        Path to the location on disk where the model is written to

    as_torchscript: bool
        If True, dump the model as a torchscript Intermediate Representation.
        If False, only write the state dict of the underlying feature extractor to disk

    **model_kwargs
        Forwarded to ```get_model_by_name```
    """
    model = get_model_by_name(name, **model_kwargs)
    if as_torchscript:
        model = torch.jit.script(model)
        model.save(out_location)
    else:
        torch.save(model.model.state_dict(), out_location)


if __name__ == "__main__":
    # compile model
    dump_model_to_disk("uni", out_location = "../models/uni.ts", as_torchscript=True, local_path = UNI_STATE_DICT_PATH)