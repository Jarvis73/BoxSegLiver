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

import numpy as np
from pathlib import Path
import scipy.ndimage as ndi
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import collections
import SimpleITK as sitk
import cv2
from functools import reduce

from utils import array_kits as arr_kits

NF_Dir = Path(__file__).parent.parent / "data/NF"
NF_Data = NF_Dir / "nii_NF"
NF_Volumes = sorted(list(NF_Data.glob("volume-*.nii.gz")), key=lambda x: str(x))
NF_Pairs = [(x, x.parent / x.name.replace("volume", "segmentation")) for x in NF_Volumes]


def nf_stat():
    disc = ndi.generate_binary_structure(3, 2)
    info = collections.defaultdict(list)
    for vol_file, seg_file in NF_Pairs:
        print(vol_file.name)
        itk_vol = sitk.ReadImage(str(vol_file))
        arr_vol = sitk.GetArrayFromImage(itk_vol)
        max_val = arr_vol.max()
        if max_val > 4000:  # remove white line
            arr_vol[arr_vol == max_val] = 0
        mean, std = arr_vol.mean(), arr_vol.std()
        itk_seg = sitk.ReadImage(str(seg_file))
        arr_seg = sitk.GetArrayFromImage(itk_seg)
        arr_seg = np.clip(arr_seg, 0, 1)
        labeled_seg, num_nf = ndi.label(arr_seg, disc)
        objs = ndi.find_objects(labeled_seg)
        size_xyz = itk_vol.GetSize()
        spacing_xyz = itk_vol.GetSpacing()
        for i, obj in enumerate(objs):
            info["PID"].append(int(vol_file.name[7:10]))
            info["NF_ID"].append(i)
            nf = labeled_seg == i + 1
            info["size/px"].append(np.count_nonzero(nf))
            bbox = arr_kits.slices_to_bbox(obj)
            info["x1"].append(bbox[2])
            info["y1"].append(bbox[1])
            info["z1"].append(bbox[0])
            info["x2"].append(bbox[5])
            info["y2"].append(bbox[4])
            info["z2"].append(bbox[3])
            info["thick"].append(bbox[3] - bbox[0])
            info["mean"].append(mean)
            info["std"].append(std)
            info["sz_x"].append(size_xyz[0])
            info["sz_y"].append(size_xyz[1])
            info["sz_z"].append(size_xyz[2])
            info["sp_x"].append(spacing_xyz[0])
            info["sp_y"].append(spacing_xyz[1])
            info["sp_z"].append(spacing_xyz[2])
    pd.DataFrame(data=info).to_csv(str(NF_Dir / "nf_stat.csv"))


def intensity_hist():
    dst_Dir = NF_Dir / "hist"
    dst_Dir.mkdir(parents=True, exist_ok=True)
    for vol_file, seg_file in NF_Pairs:
        print(seg_file.name)
        itk_vol = sitk.ReadImage(str(vol_file))
        arr_vol = sitk.GetArrayFromImage(itk_vol)
        itk_seg = sitk.ReadImage(str(seg_file))
        arr_seg = sitk.GetArrayFromImage(itk_seg)
        arr_msk = arr_vol * np.clip(arr_seg, 0, 1)
        plt.hist(arr_vol[arr_vol > 0], bins=100, range=(20, 1200), alpha=0.8, density=True)
        plt.hist(arr_msk[arr_msk > 0], bins=100, range=(20, 1200), alpha=0.8, density=True)
        plt.legend(["Body intensity", "NF intensity"])
        plt.savefig(str(dst_Dir / (seg_file.name.split(".")[0] + ".png")))
        plt.close()


def show():
    vol_file = NF_Data / "volume-027.nii.gz"
    itk_vol = sitk.ReadImage(str(vol_file))
    arr_vol = sitk.GetArrayFromImage(itk_vol)
    s = arr_vol[13].astype(np.uint16)
    mask = (s == s.max()).astype(np.uint8)
    s2 = cv2.inpaint(s, mask, 3, cv2.INPAINT_TELEA)
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(np.clip(s, 0, 600), cmap="gray")
    ax[1].imshow(np.clip(s2, 0, 600), cmap="gray")
    plt.show()


def fg_bg_prop():
    fg = 0
    bg = 0
    for _, seg_file in NF_Pairs:
        print(seg_file.name)
        itk_seg = sitk.ReadImage(str(seg_file))
        arr_seg = sitk.GetArrayFromImage(itk_seg)
        tmp = np.count_nonzero(arr_seg)
        fg += tmp
        bg += reduce(lambda a, b: a * b, arr_seg.shape, 1) - tmp
    print("fg:", fg, "bg: ", bg)


if __name__ == "__main__":
    # nf_stat()
    # show()
    # intensity_hist()
    fg_bg_prop()
