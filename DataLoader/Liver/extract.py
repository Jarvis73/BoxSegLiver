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
# noinspection PyUnresolvedReferences
import pprint
from pathlib import Path

import SimpleITK as sitk
import nibabel as nib
import numpy as np
import scipy.ndimage as ndi

from utils import array_kits


def check_dataset(file_path):
    src_path = Path(file_path)
    print("Check dataset in %s" % src_path)
    for i, case in enumerate(sorted(src_path.glob("volume-*.nii"),
                                    key=lambda x: int(str(x).split(".")[0].split("-")[-1]))):
        print("{:03d} {:47s}".format(i, str(case)), end="")
        nib_vol = nib.load(str(case))
        vh = nib_vol.header
        affine = vh.get_best_affine()
        print("{:.3f} {:.3f} {:.3f} ({:.2f})".format(affine[0, 0], affine[1, 1], affine[2, 2],
                                                     vh["pixdim"][0]), end="")
        nib_lab = nib.load(str(case.parent / case.name.replace("volume", "segmentation")))
        lh = nib_lab.header
        affine2 = lh.get_best_affine()
        print(" | {:.3f} {:.3f} {:.3f} ({:.2f})".format(affine2[0, 0], affine2[1, 1], affine2[2, 2],
                                                        lh["pixdim"][0]), end="")
        if np.allclose(affine[[0, 1, 2], [0, 1, 2]], affine2[[0, 1, 2], [0, 1, 2]], atol=1e-5):
            print(" | True")
        else:
            print(" | False")


def read_nii(file_name, out_dtype=np.int16, special=False):
    nib_vol = nib.load(str(file_name))
    vh = nib_vol.header
    affine = vh.get_best_affine()
    data = nib_vol.get_fdata().astype(out_dtype).transpose(2, 1, 0)
    if special:
        data = np.flip(data, axis=2)
    if affine[0, 0] > 0:                # Increase x from Right to Left
        data = np.flip(data, axis=2)
    if affine[1, 1] > 0:                # Increase y from Anterior to Posterior
        data = np.flip(data, axis=1)
    if affine[2, 2] < 0:                # Increase z from Interior to Superior
        data = np.flip(data, axis=0)
    return vh, data


def nii_3d_to_png(in_path, out_path):
    """ Livers in volumes 28-47 are in anatomical Left(should be Right).
        Livers in labels  28-52 are in anatomical Left(should be Right).

    Image range [-200, 250] + 200 = [0, 450]
    Save images: [0, 450] * 128     Multiply 128 for better visualization
    Save labels: [0, 2] * 64        Multiply 64 for better visualization
    """
    src_path = Path(in_path)
    dst_path = Path(out_path)
    all_meta_data = []
    disc3 = ndi.generate_binary_structure(3, connectivity=2)
    disc2 = ndi.generate_binary_structure(2, connectivity=1)
    json_file = dst_path / "meta.json"

    for i, vol_case in enumerate(sorted(src_path.glob("volume-*.nii"),
                                        key=lambda x: int(str(x).split(".")[0].split("-")[-1]))):
        print("{:03d} {:47s}".format(i, str(vol_case)))
        dst_dir = dst_path / vol_case.stem

        vh, volume = read_nii(vol_case, out_dtype=np.int16,
                              special=True if 28 <= int(vol_case.stem.split("-")[-1]) < 48 else False)
        volume = ((np.clip(volume, -200, 250) + 200) * 128).astype(np.uint16)
        lab_case = vol_case.parent / vol_case.name.replace("volume", "segmentation")
        _, labels = read_nii(lab_case, out_dtype=np.uint8,
                             special=True if 28 <= int(vol_case.stem.split("-")[-1]) < 52 else False)
        assert volume.shape == labels.shape, "Vol{} vs Lab{}".format(volume.shape, labels.shape)
        # print(vh)

        b = array_kits.extract_region(labels).tolist()
        bbox = [b[2], b[1], b[0], b[5] + 1, b[4] + 1, b[3] + 1]

        # 3D Tumor Information
        tumors, n_obj = ndi.label(labels == 2, disc3)
        slices = ndi.find_objects(tumors)
        objects = [[z.start, y.start, x.start, z.stop, y.stop, x.stop] for z, y, x in slices]
        all_centers, all_stddevs = [], []
        tumor_areas = []
        for j, sli in enumerate(slices):
            region = labels[sli] == 2
            center, stddev = array_kits.compute_robust_moments(region, index="ij", min_std=0.)
            center[0] += objects[j][0]
            center[1] += objects[j][1]
            center[2] += objects[j][2]
            all_centers.append(center.tolist())
            all_stddevs.append([round(x, 3) for x in stddev])
            tumor_areas.append(np.count_nonzero(region))

        # 2D Tumor Information
        tumor_slices_indices = np.where(np.max(labels, axis=(1, 2)) == 2)[0].tolist()
        tumor_slices = []
        tumor_slices_from_to = [0]
        tumor_slices_centers, tumor_slices_stddevs = [], []
        tumor_slices_areas = []
        start = 0
        for j in tumor_slices_indices:
            slice_ = labels[j]
            tumors_, n_obj_ = ndi.label(slice_ == 2, disc2)
            slices_ = ndi.find_objects(tumors_)
            objects_ = [[y.start, x.start, y.stop, x.stop] for y, x in slices_]
            tumor_slices.extend(objects_)
            tumor_slices_from_to.append(start + n_obj_)
            start += n_obj_
            for k, sli_ in enumerate(slices_):
                region_ = slice_[sli_] == 2
                center_, stddev_ = array_kits.compute_robust_moments(region_, index="ij", min_std=0.)
                center_[0] += objects_[k][0]
                center_[1] += objects_[k][1]
                tumor_slices_centers.append(center_.tolist())
                tumor_slices_stddevs.append([round(x, 3) for x in stddev_])
                tumor_slices_areas.append(np.count_nonzero(region_))

        meta_data = {"PID": i,
                     "vol_case": str(vol_case),
                     "lab_case": str(lab_case),
                     "size": [int(x) for x in vh.get_data_shape()[::-1]],
                     "spacing": [float(x) for x in vh.get_zooms()[::-1]],
                     "bbox": bbox,
                     "tumors": objects,
                     "tumor_areas": tumor_areas,
                     "tumor_centers": all_centers,
                     "tumor_stddevs": all_stddevs,
                     "tumor_slices_from_to": tumor_slices_from_to,
                     "tumor_slices": tumor_slices,
                     "tumor_slices_index": tumor_slices_indices,
                     "tumor_slices_centers": tumor_slices_centers,
                     "tumor_slices_stddevs": tumor_slices_stddevs,
                     "tumor_slices_areas": tumor_slices_areas}
        all_meta_data.append(meta_data)

        dst_dir.mkdir(parents=True, exist_ok=True)
        labels = labels * 64
        for j, (img, lab) in enumerate(zip(volume, labels)):
            out_img_file = dst_dir / "{:03d}_im.png".format(j)
            out_img = sitk.GetImageFromArray(img)
            sitk.WriteImage(out_img, str(out_img_file))
            out_lab_file = dst_dir / "{:03d}_lb.png".format(j)
            out_lab = sitk.GetImageFromArray(lab)
            sitk.WriteImage(out_lab, str(out_lab_file))

    with json_file.open("w") as f:
        json.dump(all_meta_data, f)


if __name__ == "__main__":
    # data_dir = "D:/Dataset/LiTS/Training_Batch"
    data_dir = Path(__file__).parent.parent.parent / "data/LiTS/Training_Batch"
    check_dataset(data_dir)
    go_on = input("Continue? [Y/n] ")
    if not go_on or go_on in ["Y", "y", "Yes", "yes"]:
        # png_dir = "D:/Dataset/LiTS/png_lits"
        png_dir = Path(__file__).parent.parent.parent / "data/LiTS/png"
        nii_3d_to_png(data_dir, png_dir)
        # json_file = Path(png_dir) / "meta.json"
        # with json_file.open() as f:
        #     d = json.load(f)
        # pprint.pprint(d)
