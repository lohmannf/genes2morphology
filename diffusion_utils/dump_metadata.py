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
import json
import argparse
import PIL
import numpy as np
import pandas as pd

def make_metadata_file(args):
    '''
    Create a metadata.jsonl file that assigns each file in an image folder to an index in the corresponding gex array.
    Enables use of HuggingFace datasets with Image-Gene pairs.
    '''
    files = sorted(os.listdir(args.image_dir))
    files = [f for f in files if f.split(".")[-1].lower() == "png"]
    prompts = range(len(files))

    metadata = pd.DataFrame({"file_name": files, "gex_emb": prompts})
    metadata.to_csv(os.path.join(args.image_dir, "metadata.csv"), index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Create Metadata File in correct format")
    parser.add_argument("--image-dir", type=str, required=True, help="Path to image directory with lexicographically ordered files.")
    args = parser.parse_args()
    make_metadata_file(args)