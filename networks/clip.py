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
import pytorch_lightning as pl
import numpy as np
import torch.nn.functional as F
from torch import nn
from omegaconf import OmegaConf

from networks import histo_models

CLIP_PATH = "../models/clip_dict.pt" # path to pretrained CLIP weights

class ProjectionHead(nn.Module):
    # Ported from https://github.com/moein-shariatnia/OpenAI-CLIP
    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        dropout: float,
        residual: bool = True,
        norm_latent: bool = True
    ):
        super().__init__()
        self.projection = nn.Linear(input_dim, latent_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(latent_dim, latent_dim)
        self.dropout = nn.Dropout(dropout)
        self.latent_dim = latent_dim

        if norm_latent:
            self.layer_norm = nn.LayerNorm(latent_dim)
        else:
            self.layer_norm = False

        self.residual = residual
    
    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        
        if self.residual:
            x = x + projected

        if self.layer_norm:
            x = self.layer_norm(x)
        return x
    
class LinearBlock(nn.Module):
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 dropout: float):
        super().__init__()

        self.l1 = nn.Linear(input_dim, input_dim)
        self.act = nn.GELU()
        self.l2 = nn.Linear(input_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.l1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.l2(x)
        x = self.act(x)
        return x
    
def clip_loss(emb1, emb2, temperature: float = 1.):
    emb1 = F.normalize(emb1, dim=-1)
    emb2 = F.normalize(emb2, dim=-1)

    logits = (emb2 @ emb1.T) / temperature
    
    targets = torch.arange(len(emb1)).to(logits).to(torch.int64)
    ce1 = F.cross_entropy(logits, targets, reduction='none')
    ce2 = F.cross_entropy(logits.T, targets.T, reduction='none')
    loss =  (ce1 + ce2) / 2.0

    return loss.mean()

class CLIP(pl.LightningModule):
    '''
    A model that learns an aligned latent representation of image and gene expression
    using a contastive loss.
    '''
    def __init__(self, ckpt_file=None, cfg=None, trainable=False):
        '''
        Parameters
        ----------
        cfg: OmegaConf | None
            Config containing hyperparameters for model and training.
            Can be omitted when loading the model from a checkpoint

        ckpt_file: str | None
            Path to a dict-formatted checkpoint file

        trainable: bool
            Whether the model parameters are trainable.
            Does not apply to the UNI image feature extractor, which always remains frozen
        '''
        super().__init__()

        if trainable and (cfg is None):
            raise ValueError("Cannot train model without training hyperparameters, please provide cfg")
        
        if cfg is None:
            self.cfg = OmegaConf.create(dict(image=dict(), gex=dict()))
        else:
            self.cfg = cfg
        
        if ckpt_file is not None:
            self.load_weights(ckpt_file)
        else:
            self.img_encoder = nn.Sequential(
                histo_models.get_model_by_name("uni", local_path = histo_models.UNI_STATE_DICT_PATH),
                ProjectionHead(**OmegaConf.to_object(cfg.image.projection))
            )
            self.gex_encoder = nn.Sequential(
                LinearBlock(**OmegaConf.to_object(cfg.gex.kwargs)),
                ProjectionHead(**OmegaConf.to_object(cfg.gex.projection))
            )
            self.latent_dim = self.cfg.latent_dim

        if not trainable:
            self.eval()
            for p in self.parameters():
                p.requires_grad_(False)

        self.val_loss = []

    def load_weights(self, ckpt_file: str):
        """
        Load a dict-formatted checkpoint into the model.
        In contrast to the naive `load_from_checkpoint` inherited from pl, this method
        works across different PyTorch versions.

        Parameters
        ------------
        ckpt_file: str
            Path to the dict-style checkpoint formatted with `convert_ckpt_to_dict`

        Returns
        -------
        None
        """
        ckpt = torch.load(ckpt_file, map_location="cpu")

        # update the config
        self.cfg.image.projection = ckpt["image"]["projection_kwargs"]
        self.cfg.gex.kwargs = ckpt["gex"]["encoder_kwargs"]
        self.cfg.gex.projection = ckpt["gex"]["projection_kwargs"]

        # init modules and weights
        self.img_encoder = nn.Sequential(
            histo_models.get_model_by_name("uni", local_path = histo_models.UNI_STATE_DICT_PATH),
            ProjectionHead(**ckpt["image"]["projection_kwargs"])
        )
        self.gex_encoder = nn.Sequential(
            LinearBlock(**ckpt["gex"]["encoder_kwargs"]),
            ProjectionHead(**ckpt["gex"]["projection_kwargs"])
        )
        self.img_encoder[1].load_state_dict(ckpt["image"]["projection_state_dict"], strict=True)
        self.gex_encoder[0].load_state_dict(ckpt["gex"]["encoder_state_dict"], strict=True)
        self.gex_encoder[1].load_state_dict(ckpt["gex"]["projection_state_dict"], strict=True)
        self.latent_dim = ckpt["image"]["projection_kwargs"]["latent_dim"]
    
    @staticmethod
    def convert_ckpt_to_dict(ckpt_file: str, cfg: OmegaConf, out_location: str):
        """
        Convert a pl checkpoint to a dict-style checkpoint containing only
        kwargs and state dicts

        Parameters
        ----------
        ckpt_file: str
            Path to the pl checkpoint created during training

        out_location: str
            Path to the output file

        cfg: OmegaConf | str
            The config containing hyperparameters of the model
        
        Returns
        ------
        None
        """
        ckpt = torch.load(ckpt_file, map_location="cpu")
        if isinstance(cfg, str):
            cfg = OmegaConf.load(cfg)
        model = CLIP(cfg=cfg)
        model.load_state_dict(ckpt["state_dict"])

        dict_ckpt = dict(image=dict(), gex=dict())
        dict_ckpt["image"]["projection_kwargs"] = OmegaConf.to_object(model.cfg.image.projection)
        dict_ckpt["image"]["projection_state_dict"] = model.img_encoder[1].state_dict()
        dict_ckpt["gex"]["encoder_kwargs"] = OmegaConf.to_object(model.cfg.gex.kwargs)
        dict_ckpt["gex"]["encoder_state_dict"] = model.gex_encoder[0].state_dict()
        dict_ckpt["gex"]["projection_kwargs"] = OmegaConf.to_object(model.cfg.gex.projection)
        dict_ckpt["gex"]["projection_state_dict"] = model.gex_encoder[1].state_dict()

        torch.save(ckpt, out_location)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def encode_image(self, images: torch.Tensor, div255: bool = False) -> torch.Tensor:
        if div255:
            images = images.to(torch.float32) / 255.
        image_features = self.img_encoder(images)
        image_features = F.normalize(image_features, dim=-1)
        return image_features

    def encode_gex(self, gex: torch.Tensor) -> torch.Tensor:
        gex_features = self.gex_encoder(gex)
        gex_features = F.normalize(gex_features, dim=-1)
        return gex_features

    def forward(self, images: torch.Tensor, gex: torch.Tensor, div255: bool = False) -> torch.Tensor:
        if gex is None:
            return self.encode_image(images, div255)
        if images is None:
            return self.encode_gex(gex)
        assert len(images) == len(gex)
        image_features = self.encode_image(images, div255)
        gex_features = self.encode_gex(gex)
        joint_features = torch.cat([image_features, gex_features], 1)
        return joint_features
    
    ### PYTORCH LIGHTNING METHODS ###

    def training_step(self, batch, batch_idx):
        embs_img = self.encode_image(batch[0], div255=False)
        embs_gex = self.encode_gex(batch[1])

        loss = clip_loss(embs_img, embs_gex)
        self.log("train/loss", loss.item())
        return loss

    def configure_optimizers(self):
        params = [
            {"params": self.img_encoder.parameters()}, 
            {"params": self.gex_encoder.parameters()}]

        if self.cfg.optimizer.name == "adamw":
            optim = torch.optim.AdamW(params, lr=self.cfg.optimizer.lr, weight_decay=self.cfg.optimizer.decay)
        elif self.cfg.optimizer.name == "adam":
            optim = torch.optim.Adam(params, lr=self.cfg.optimizer.lr, weight_decay=self.cfg.optimizer.decay)
        elif self.cfg.optimizer.name == "sgd":
            optim = torch.optim.SGD(params, lr=self.cfg.optimizer.lr, weight_decay=self.cfg.optimizer.decay, momentum=self.cfg.optimizer.momentum)
        elif self.cfg.optimizer.name == "rmsprop":
            optim = torch.optim.RMSprop(params, lr=self.cfg.optimizer.lr, weight_decay=self.cfg.optimizer.decay)
        return optim
    
    def validation_step(self, batch, batch_idx):
        embs_img = self.encode_image(batch[0], div255=False)
        embs_gex = self.encode_gex(batch[1])

        loss = clip_loss(embs_img, embs_gex)
        self.val_loss.append(loss.item())

    def test_step(self, batch, batch_idx):
        self.validation_step(batch, batch_idx)
    
    def on_validation_epoch_end(self):
        self.log("val/loss", np.mean(self.val_loss))
        self.val_loss.clear()

    def on_test_epoch_end(self):
        self.log("test/loss", np.mean(self.val_loss))
        self.val_loss.clear()

    def on_sanity_check_end(self):
        # log validation metrics before starting training
        self.on_validation_epoch_end()