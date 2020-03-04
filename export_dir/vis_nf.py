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

import cv2
import json
import numpy as np
import tensorflow as tf
from pathlib import Path
from pprint import pprint
from DataLoader.Liver import nii_kits
from utils import array_kits


def dump_img(pb_model, vol_path, meta=None, slices="all"):
    inputs_dir = pb_model / "inputs_npy"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    vol_path = Path(vol_path)
    _, img = nii_kits.read_nii(vol_path)
    img = np.clip(img, 0, 900) / 900
    img = img.astype(np.float32).transpose((1, 2, 0))

    if meta is None:
        meta = Path(__file__).parent.parent / "DataLoader/NF/prepare/meta.json"
    else:
        meta = Path(meta)
    with meta.open() as f:
        meta_data = json.load(f)
    case = meta_data[int(vol_path.name.split(".")[0].split("-")[-1])]
    if slices == "all":
        slices = list(range(0, case["size"][0]))

    if img.shape[-1] > 512:
        res = []
        for i in range(0, img.shape[-1], 512):
            res.append(cv2.resize(img[:, :, i:i + 512], (320, 960), interpolation=cv2.INTER_LINEAR))
        img = np.concatenate(res, axis=-1)
    else:
        img = cv2.resize(img, (320, 960), interpolation=cv2.INTER_LINEAR)

    for i in slices:
        save_path = str(inputs_dir / (vol_path.name.split(".")[0] + "-{}.npy".format(i)))
        if i == 0:
            np.save(save_path, np.concatenate((np.zeros((960, 320, 1), dtype=img.dtype), img[:, :, :2]), axis=-1)[None])
        elif i == case["size"][0] - 1:
            np.save(save_path, np.concatenate((img[:, :, :2], np.zeros((960, 320, 1), dtype=img.dtype)), axis=-1)[None])
        else:
            np.save(save_path, img[:, :, i - 1:i + 2][None])


def gen_guide(save_path, shape=(960, 320), guide_path=None, meta=None, guide_volume=None, guide_slice=None, inten=1.):
    if guide_path is not None:
        if meta is None:
            meta = Path(__file__).parent.parent / "DataLoader/NF/prepare/meta.json"
        else:
            meta = Path(meta)
        with meta.open() as f:
            meta_data = json.load(f)
        case = [x for x in meta_data if x["PID"] == guide_volume]

        with Path(guide_path).open() as f:
            gd_meta = json.load(f)
        centers = np.array(gd_meta[str(guide_volume)][str(guide_slice)]["centers"])
        stddevs = np.array(gd_meta[str(guide_volume)][str(guide_slice)]["stddevs"])
        g = array_kits.create_gaussian_distribution_v2(case[0]["size"][1:], centers, stddevs)
        g = cv2.resize(g, shape[::-1], cv2.INTER_LINEAR)
        g = g[None, ..., None] / 2 * inten + 0.5
    else:
        g = np.ones((1,) + shape + (1,), np.float32) * 0.5

    np.save(save_path, g)


def save(num, sli):
    volpath = Path(r"D:\0WorkSpace\MedicalImageSegmentation\data\NF\nii_NF\volume-{:03d}.nii.gz".format(num))
    dump_img(pbmodel, volpath, slices=[sli])
    save_path = pbmodel / "inputs_npy/guide-{}-{}-v0.npy".format(num, sli)
    gen_guide(save_path)
    save_path = pbmodel / "inputs_npy/guide-{}-{}-v1.npy".format(num, sli)
    guide_path = pbmodel / "inter-{}-{}.json".format(num, sli)
    gen_guide(save_path, guide_path=guide_path, guide_volume=num, guide_slice=sli, inten=2.)


if __name__ == "__main__":
    pbmodel = Path("E:/Temp/111_nf_sp_rand")

    # volpath = Path(r"D:\0WorkSpace\MedicalImageSegmentation\data\NF\nii_NF\volume-036.nii.gz")
    # dump_img(pbmodel, volpath, slices=[11])
    # save_path = pbmodel / "inputs_npy/guide-v0.npy"
    # gen_guide(save_path)
    # save_path = pbmodel / "inputs_npy/guide-v1.npy"
    # guide_path = pbmodel / "inter.json"
    # gen_guide(save_path, guide_path=guide_path, guide_volume=36, guide_slice=11)

    save(87, 10)
