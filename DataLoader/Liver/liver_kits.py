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
import shutil
import multiprocessing
import numpy as np
from pathlib import Path
from DataLoader.Liver import nii_kits
from utils import array_kits
from skimage.transform import resize

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


def merge_volumes_wrap(tag_list, out_dir):
    p = multiprocessing.Pool(4)
    source = [ROOT_DIR / "model_dir" / tag / "prediction" for tag in tag_list]
    out_dir = ROOT_DIR / "model_dir" / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    p.map(merge_volumes, zip(range(70), [source] * 70, [out_dir] * 70))


def merge_volumes(args_):
    i, source, out_dir = args_
    print("Process", i)
    all_volume = []
    for src in source:
        case = src / "{}.npz".format(i)
        all_volume.append(np.load(str(case))['arr_0'])
    mean_volume = np.mean(all_volume, axis=0)
    mean_volume = np.argmax(mean_volume, axis=-1)
    liver = (mean_volume == 1).astype(np.uint8)
    tumor = (mean_volume == 2).astype(np.uint8)

    # Add tumor to liver volume
    liver += tumor
    # Find largest component --> for liver
    liver = array_kits.get_largest_component(liver, rank=3)
    # Remove false positives outside liver region
    tumor *= liver
    # Add to one volume: liver=1, tumor=2
    final_volume = liver + tumor

    volume_file = ROOT_DIR / "data/LiTS/Test_Batch/test-volume-{}.nii".format(i)
    vh = nii_kits.read_nii(volume_file, only_header=True)
    save_file = out_dir / "test-segmentation-{}.nii".format(i)
    nii_kits.write_nii(final_volume, vh, save_file, out_dtype=np.uint8)


def fix_test_59(out_dir):
    out_dir = ROOT_DIR / "model_dir" / out_dir
    bk_dir = out_dir / "backup"
    bk_dir.mkdir(parents=True, exist_ok=True)
    shutil.move(str(out_dir / "test-segmentation-59.nii"), str(bk_dir))
    vh, v = nii_kits.read_nii(bk_dir / "test-segmentation-59.nii", out_dtype=np.uint8)
    vh["dim"][3] = 79
    vh["pixdim"][3] = 3.
    vh["srow_y"][2] = -3.
    v2 = resize(v, (512, 79, 512), order=0, preserve_range=True)
    nii_kits.write_nii(v2, vh, out_dir / "test-segmentation-59.nii", out_dtype=np.uint8)


if __name__ == "__main__":
    # tag_list = ["002_unet_liver", "002_unet_liver_f0"]
    # out_dir = "merge_002_unet_liver"
    # merge_liver(tag_list, out_dir)
    # -----------------------------------------------------------------
    # json_file = "./DataLoader/Liver/prepare/test_meta.json"
    # liver_pattern = "./model_dir/merge_002_unet_liver/liver-*.nii.gz"
    # update_json_with_liver(json_file, liver_pattern)
    # -----------------------------------------------------------------
    # tag_list = ["001_unet_noise_0_05_f0",
    #             "001_unet_noise_0_05_f1",
    #             "001_unet_noise_0_05",
    #             "001_unet_noise_0_05_f3",
    #             "001_unet_noise_0_05_f4"]
    # out_dir = "merge_001_unet_noise_0_05"
    # -----------------------------------------------------------------
    tag_list = ["013_gnet_sp_rand_f0",
                "013_gnet_sp_rand_f1",
                "013_gnet_sp_rand",
                "013_gnet_sp_rand_f3",
                "013_gnet_sp_rand_f4"]
    out_dir = "merge_013_gnet_sp_rand"
    merge_volumes_wrap(tag_list, out_dir)
    # -----------------------------------------------------------------
    fix_test_59(out_dir)
