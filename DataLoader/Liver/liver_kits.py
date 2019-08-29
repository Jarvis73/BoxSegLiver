# Copyright 2019 Jianwei Zhang All Right Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# =================================================================================

import json
import numpy as np
from pathlib import Path
from DataLoader.Liver import nii_kits
from utils import array_kits

ROOT_DIR = Path(__file__).parent.parent.parent


def update_json_with_liver(json_file, liver_pattern):
    json_file = Path(json_file)
    with json_file.open() as f:
        meta = json.load(f)

    liver_pattern = Path(liver_pattern)
    for liver_file in liver_pattern.parent.glob(liver_pattern.name):
        print(liver_file)
        pid = int(liver_file.name.split(".")[0].split("-")[-1])
        vh, volume = nii_kits.read_nii(liver_file, out_dtype=np.uint8)
        x1, y1, z1, x2, y2, z2 = array_kits.bbox_from_mask(volume, 1).tolist()
        assert meta[pid]["PID"] == pid
        meta[pid]["bbox"] = [z1, y1, x1, z2 + 1, y2 + 1, x2 + 1]

    new_json_file = json_file.parent / (json_file.stem + "_update.json")
    with new_json_file.open("w") as f:
        json.dump(meta, f)


def merge_liver(tag_list, out_dir):
    source = [ROOT_DIR / "model_dir" / tag / "prediction" for tag in tag_list]
    out_dir = ROOT_DIR / "model_dir" / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    for i in range(70):
        print("Process", i)
        all_livers = []
        for src in source:
            case = src / "{}.npz".format(i)
            all_livers.append(np.load(str(case))['arr_0'])
        mean_liver = np.mean(all_livers, axis=0)
        mean_liver = np.argmax(mean_liver, axis=-1)
        volume_file = ROOT_DIR / "data/LiTS/Test_Batch/test-volume-{}.nii".format(i)
        vh = nii_kits.read_nii(volume_file, only_header=True)
        save_file = out_dir / "liver-{}.nii.gz".format(i)
        nii_kits.write_nii(mean_liver, vh, save_file)


if __name__ == "__main__":
    # tag_list = ["002_unet_liver", "002_unet_liver_f0"]
    # out_dir = "merge_002_unet_liver"
    # merge_liver(tag_list, out_dir)
    # -----------------------------------------------------------------
    json_file = "./DataLoader/Liver/prepare/test_meta.json"
    liver_pattern = "./model_dir/merge_002_unet_liver/liver-*.nii.gz"
    update_json_with_liver(json_file, liver_pattern)

