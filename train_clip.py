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

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from omegaconf import OmegaConf
import torch
import numpy as np
import random
import hydra
import wandb
import torchvision.transforms as tf
import scanpy as sc

from networks.clip import CLIP
from training.data_zip import ImageGeneDataset

def get_transforms(train: bool = False):
    transforms = [tf.Lambda(lambda x: torch.clamp(x.float() / 255., 0, 1))]
    if train:
        transforms.extend([
            tf.RandomHorizontalFlip(0.5),
            tf.RandomVerticalFlip(0.5),
            tf.ColorJitter(brightness = 0.2, contrast = 0.2, saturation = 0.2, hue = 0.2),
        ])
    return tf.Compose(transforms)

@hydra.main(version_base=None, config_path="./assets", config_name = "clip_config")
def main(cfg):
    # seed for reproducibility
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    # setup W&B logging
    logger = WandbLogger(project = cfg.wandb.project)
    logger.experiment.config.update(OmegaConf.to_container(cfg))
    model = CLIP(cfg=cfg, trainable=True)

    # Load data for training and validation
    print("Loading training and validation data")
    train_subset = np.random.choice(sc.read_h5ad(cfg.data.train_gex).n_obs, cfg.data.max_size, replace=False)
    train_set = ImageGeneDataset(path = cfg.data.train_image_dir, 
                                    gex_path = cfg.data.train_gex, 
                                    mask = train_subset, 
                                    use_labels = True, 
                                    resolution = 128)
    val_set = ImageGeneDataset(path = cfg.data.val_image_dir, 
                                gex_path = cfg.data.val_gex, 
                                use_labels = True,
                                resolution = 128)

    train_transform = get_transforms(train = True)
    val_transform = get_transforms()

    def get_collate_fn(transform):
        def collate(batch):
            imgs = []
            gex = []
            for x in batch:
                imgs.append(transform(torch.from_numpy(x[0])))
                gex.append(torch.from_numpy(x[1]).float())
            return torch.stack(imgs, dim=0), torch.stack(gex, dim=0)
        return collate

    train_dl = torch.utils.data.DataLoader(train_set, 
                                            batch_size = cfg.training.batch_size, 
                                            shuffle = True, 
                                            drop_last = True, 
                                            num_workers = 2,
                                            collate_fn = get_collate_fn(train_transform))
    val_dl = torch.utils.data.DataLoader(val_set, 
                                            batch_size = cfg.training.val_batch_size, 
                                            shuffle = True, 
                                            num_workers = 2,
                                            collate_fn = get_collate_fn(val_transform))
    print("Finished setting up data")
    
    # Run training
    checkpoint_cbk = ModelCheckpoint(save_top_k=5, monitor="val/loss", mode="min", 
                                     save_last=True, every_n_train_steps=cfg.training.ckpt_freq)
    trainer = pl.Trainer(logger = logger,
                         accelerator="gpu",
                         precision=32,
                         devices=1, 
                         max_epochs=cfg.training.epochs,
                         limit_val_batches=50,
                         val_check_interval=cfg.training.val_freq,
                         callbacks=[checkpoint_cbk],
                         log_every_n_steps=cfg.training.log_freq
                         )
    trainer.fit(model = model, train_dataloaders=train_dl, val_dataloaders=val_dl)
    wandb.finish()
    
if __name__ == "__main__":
    main()