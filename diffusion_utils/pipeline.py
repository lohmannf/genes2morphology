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

from typing import Optional
import torch

from diffusers import StableDiffusionPipeline


class CustomStableDiffusionPipeline(StableDiffusionPipeline):
    def __init__(self,
                 vae,
                 unet,
                 scheduler,
                 feature_extractor,
                 tokenizer=None,
                 safety_checker=None,
                 image_encoder=None,
                 text_encoder=None,
                 requires_safety_checker=False):
        super().__init__(vae=vae,
                         text_encoder=None,
                         tokenizer=tokenizer,
                         unet=unet,
                         scheduler=scheduler,
                         safety_checker=safety_checker,
                         feature_extractor=feature_extractor,
                         image_encoder=image_encoder,
                         requires_safety_checker=requires_safety_checker)
        
    def encode_prompt(
        self,
        prompt,
        device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=False,
        negative_prompt=None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        lora_scale: Optional[float] = None,
        clip_skip: Optional[int] = None,
    ):  
        prompt_embeds = prompt_embeds.to(device)
        if negative_prompt_embeds is not None:
            negative_prompt_embeds = negative_prompt_embeds.repeat(prompt_embeds.shape[0], 1, 1).to(device)
        return prompt_embeds, negative_prompt_embeds