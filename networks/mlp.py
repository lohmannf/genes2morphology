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

from torch import nn
import torch.nn.functional as F
import torch

class MLPEncoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, hidden_dims: list[int],
                 input_dropout: float, dropout: float):
        super().__init__()
        self.latent_dim = latent_dim

        self.network = nn.ModuleList()
        for i in range(len(hidden_dims)):
            if i == 0:
                self.network.append(
                nn.Sequential(
                    nn.Dropout(p=input_dropout),
                    nn.Linear(input_dim, hidden_dims[i]),
                    nn.BatchNorm1d(hidden_dims[i]),
                    nn.PReLU()
                ))
            else:
                self.network.append(
                nn.Sequential(
                    nn.Dropout(p=dropout),
                    nn.Linear(hidden_dims[i-1], hidden_dims[i]),
                    nn.BatchNorm1d(hidden_dims[i]),
                    nn.PReLU()
                ))
        
        self.network.append(nn.Linear(hidden_dims[-1], latent_dim))
    
    @property
    def device(self) -> torch.device:
        return next(self.network.parameters()).device

    def forward(self, x):
        for layer in self.network:
            x = layer(x)
        x = F.normalize(x, dim=1)
        return x