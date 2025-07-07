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
import numpy as np
import os

def slerp(t, v0, v1, DOT_THRESHOLD=0.9995):
    """ helper function to spherically interpolate two arrays v1 v2 """
    # from https://gist.github.com/karpathy/00103b0037c5aaea32fe1da1af553355
    inputs_are_torch = False
    if not isinstance(v0, np.ndarray):
        inputs_are_torch = True
        input_device = v0.device
        v0 = v0.cpu().numpy()
        v1 = v1.cpu().numpy()

    dot = np.sum(v0 * v1 / (np.linalg.norm(v0) * np.linalg.norm(v1)))
    if np.abs(dot) > DOT_THRESHOLD:
        v2 = (1 - t) * v0 + t * v1
    else:
        theta_0 = np.arccos(dot)
        sin_theta_0 = np.sin(theta_0)
        theta_t = theta_0 * t
        sin_theta_t = np.sin(theta_t)
        s0 = np.sin(theta_0 - theta_t) / sin_theta_0
        s1 = sin_theta_t / sin_theta_0
        v2 = s0 * v0 + s1 * v1

    if inputs_are_torch:
        v2 = torch.from_numpy(v2).to(input_device)

    return v2

def main(prompt1, prompt2, n_steps):
    """
    Interpolate between prompt1 and prompt2 in n_steps
    """
    return torch.stack([torch.from_numpy(slerp(t, prompt1, prompt2)) for t in np.linspace(0,1, n_steps+2)])

if __name__ == "__main__":
    n_steps = 10
    out_dir = "../data/embeddings"
    os.makedirs(out_dir, exist_ok=True)

    for idx_1, idx_2 in zip([0,1,2,3,4],[1,2,3,4,0]):
        avgs = np.load("../data/space_ranger_outputs/TENX45/gene_clip_embs_avg_clusters5.npy")
        np.save(os.path.join(out_dir, f"interpolated_cluster_{idx_1}_{idx_2}"), main(avgs[idx_1], avgs[idx_2], n_steps).numpy(), allow_pickle=False)