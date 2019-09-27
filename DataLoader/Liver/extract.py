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
# noinspection PyUnresolvedReferences
import pprint
import multiprocessing
from pathlib import Path

import SimpleITK as sitk
import nibabel as nib
import numpy as np
import scipy.ndimage as ndi

from DataLoader.Liver import nii_kits
from utils import array_kits

GRAY_MIN = -250
GRAY_MAX = 300
IM_SCALE = 64
LB_SCALE = 64
ROOT_DIR = Path(__file__).parent.parent.parent


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


def process_case(args):
    vol_case, i, dst_path, only_meta = args
    print("{:03d} {:47s}".format(i, str(vol_case)))

    disc3 = ndi.generate_binary_structure(3, connectivity=2)
    # disc2 = ndi.generate_binary_structure(2, connectivity=1)
    dst_dir = dst_path / vol_case.stem
    pid = int(vol_case.stem.split("-")[-1])

    vh, volume = nii_kits.read_nii(vol_case, out_dtype=np.int16,
                                   special=True if 28 <= int(vol_case.stem.split("-")[-1]) < 48 else False)
    volume = ((np.clip(volume, GRAY_MIN, GRAY_MAX) - GRAY_MIN) * IM_SCALE).astype(np.uint16)
    lab_case = vol_case.parent / vol_case.name.replace("volume", "segmentation")
    _, labels = nii_kits.read_nii(lab_case, out_dtype=np.uint8,
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
    bbox_map_3dto2d = {i: {"centers": [], "stddevs": [], "areas": [], "slices": [], "z": []}
                       for i in range(len(slices))}
    z_rev_map = {i: {"tid": [], "rid": []} for i in range(volume.shape[0])}
    for j, sli in enumerate(slices):
        region = labels[sli] == 2
        center, stddev = array_kits.compute_robust_moments(region, indexing="ij", min_std=0.)
        center[0] += objects[j][0]
        center[1] += objects[j][1]
        center[2] += objects[j][2]
        all_centers.append(center.tolist())
        all_stddevs.append([round(x, 3) for x in stddev])
        tumor_areas.append(np.count_nonzero(region))
        for k in range(region.shape[0]):
            patch = region[k]
            center_, stddev_ = array_kits.compute_robust_moments(patch, indexing="ij", min_std=0.)
            center_[0] += objects[j][1]
            center_[1] += objects[j][2]
            bbox_map_3dto2d[j]["centers"].append(center_.tolist())
            bbox_map_3dto2d[j]["stddevs"].append([round(x, 3) for x in stddev_])
            bbox_map_3dto2d[j]["areas"].append(np.count_nonzero(patch))
            x1, y1, x2, y2 = array_kits.bbox_from_mask(patch, mask_values=1).tolist()
            bbox_map_3dto2d[j]["slices"].append([y1 + objects[j][1], x1 + objects[j][2],
                                                 y2 + 1 + objects[j][1], x2 + 1 + objects[j][2]])
            bbox_map_3dto2d[j]["z"].append(objects[j][0] + k)
            z_rev_map[objects[j][0] + k]["tid"].append(j)
            z_rev_map[objects[j][0] + k]["rid"].append(k)

    tumor_slices_indices = [j for j in z_rev_map if len(z_rev_map[j]["tid"]) > 0]
    tumor_slices_from_to = [0]
    tumor_slices_centers = []
    tumor_slices_stddevs = []
    tumor_slices_areas = []
    tumor_slices = []
    tumor_slices_tid = []
    start = 0
    for j in tumor_slices_indices:
        length = len(z_rev_map[j]["tid"])
        tumor_slices_from_to.append(length + start)
        start += length
        for k in range(len(z_rev_map[j]["tid"])):
            tid = z_rev_map[j]["tid"][k]
            rid = z_rev_map[j]["rid"][k]
            tumor_slices_centers.append(bbox_map_3dto2d[tid]["centers"][rid])
            tumor_slices_stddevs.append(bbox_map_3dto2d[tid]["stddevs"][rid])
            tumor_slices_areas.append(bbox_map_3dto2d[tid]["areas"][rid])
            tumor_slices.append(bbox_map_3dto2d[tid]["slices"][rid])
            tumor_slices_tid.append(tid)

    # 2D Tumor Information
    # tumor_slices_indices = np.where(np.max(labels, axis=(1, 2)) == 2)[0].tolist()
    # tumor_slices = []
    # tumor_slices_from_to = [0]
    # tumor_slices_centers, tumor_slices_stddevs = [], []
    # tumor_slices_areas = []
    # start = 0
    # for j in tumor_slices_indices:
    #     slice_ = labels[j]
    #     tumors_, n_obj_ = ndi.label(slice_ == 2, disc2)
    #     slices_ = ndi.find_objects(tumors_)
    #     objects_ = [[y.start, x.start, y.stop, x.stop] for y, x in slices_]
    #     tumor_slices.extend(objects_)
    #     tumor_slices_from_to.append(start + n_obj_)
    #     start += n_obj_
    #     for k, sli_ in enumerate(slices_):
    #         region_ = slice_[sli_] == 2
    #         center_, stddev_ = array_kits.compute_robust_moments(region_, indexing="ij", min_std=0.)
    #         center_[0] += objects_[k][0]
    #         center_[1] += objects_[k][1]
    #         tumor_slices_centers.append(center_.tolist())
    #         tumor_slices_stddevs.append([round(x, 3) for x in stddev_])
    #         tumor_slices_areas.append(np.count_nonzero(region_))

    meta_data = {"PID": pid,
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
                 "tumor_slices_areas": tumor_slices_areas,
                 "tumor_slices_tid": tumor_slices_tid}

    if not only_meta:
        dst_dir.mkdir(parents=True, exist_ok=True)
        labels = labels * LB_SCALE
        for j, (img, lab) in enumerate(zip(volume, labels)):
            out_img_file = dst_dir / "{:03d}_im.png".format(j)
            out_img = sitk.GetImageFromArray(img)
            sitk.WriteImage(out_img, str(out_img_file))
            out_lab_file = dst_dir / "{:03d}_lb.png".format(j)
            out_lab = sitk.GetImageFromArray(lab)
            sitk.WriteImage(out_lab, str(out_lab_file))

    return meta_data


def nii_3d_to_png(in_path, out_path, only_meta=False):
    """ Livers in volumes 28-47 are in anatomical Left(should be Right).
        Livers in labels  28-52 are in anatomical Left(should be Right).

    Image range [-250, 300] + 250 = [0, 550]
    Save images: [0, 550] * 64     Multiply 64 for better visualization
    Save labels: [0, 2] * 64       Multiply 64 for better visualization
    """
    src_path = Path(in_path)
    dst_path = Path(out_path)
    json_file = dst_path / "meta.json"

    all_files = sorted(src_path.glob("volume-*.nii"), key=lambda x: int(str(x).split(".")[0].split("-")[-1]))
    p = multiprocessing.Pool(4)
    all_meta_data = p.map(process_case,
                          zip(all_files, range(len(all_files)), [dst_path] * len(all_files),
                              [only_meta] * len(all_files)))
    all_meta_data = sorted(all_meta_data, key=lambda x: x["PID"])

    with json_file.open("w") as f:
        json.dump(all_meta_data, f)


def run_nii_3d_to_png():
    # data_dir = "D:/Dataset/LiTS/Training_Batch"
    data_dir = Path(__file__).parent.parent.parent / "data/LiTS/Training_Batch"
    check = input("Check dataset? [y/N]")
    if check in ["Y", "y", "Yes", "yes"]:
        check_dataset(data_dir)
    go_on = input("Continue? [Y/n] ")
    if not go_on or go_on in ["Y", "y", "Yes", "yes"]:
        only_meta = input("Only meta? [Y/n]")
        if not only_meta or only_meta in ["Y", "y", "Yes", "yes"]:
            only_meta = True
        else:
            only_meta = False
        # png_dir = "D:/Dataset/LiTS/png_lits"
        png_dir = Path(__file__).parent.parent.parent / "data/LiTS/png"
        nii_3d_to_png(data_dir, png_dir, only_meta=only_meta)
        # json_file = Path(png_dir) / "meta.json"
        # with json_file.open() as f:
        #     d = json.load(f)
        # pprint.pprint(d)


def dump_hist_feature(in_path, out_path,
                      mode="train",
                      bins=100,
                      xrng=(GRAY_MIN, GRAY_MAX),
                      number=0):
    """
    Compute histogram features for context guide

    Parameters
    ----------
    in_path: str
        data directory
    out_path: str
        npy/npz directory
    mode: str
        for train or eval
    bins: int
        number of bins of histogram, default 100
    xrng: tuple
        (x_min, x_max), default (GRAY_MIN, GRAY_MAX)
    number: int
        for debug

    Returns
    -------

    """
    print("\n\nDeprecated!!!!!!!!!!!!!!!!!! Use v2 !!!!!!!!!!!!!\n\n")
    src_path = Path(in_path)
    dst_path = Path(out_path) / mode
    dst_path.mkdir(parents=True, exist_ok=True)
    dst_file = str(dst_path / "%03d")

    for i, vol_case in enumerate(sorted(src_path.glob("volume-*.nii"),
                                        key=lambda x: int(str(x).split(".")[0].split("-")[-1]))):
        if number >= 0 and number != i:
            continue
        PID = int(vol_case.stem.split("-")[-1])

        vh, volume = nii_kits.read_lits(vol_case.stem.split("-")[-1], "vol", vol_case)
        lab_case = vol_case.parent / vol_case.name.replace("volume", "segmentation")
        _, labels = nii_kits.read_lits(vol_case.stem.split("-")[-1], "lab", lab_case)
        assert volume.shape == labels.shape, "Vol{} vs Lab{}".format(volume.shape, labels.shape)

        if mode == "train":
            tumor_labels = labels
        else:
            tumor_labels = array_kits.get_guide_image(
                labels, obj_val=2, guide="middle", tile_guide=True) * 2

        slice_hists = np.empty((volume.shape[0], bins * 2), dtype=np.float32)
        for k in range(volume.shape[0]):
            with np.errstate(invalid='ignore'):
                val1, _ = np.histogram(volume[k][labels[k] >= 1], bins=bins, range=xrng, density=True)
                val2, _ = np.histogram(volume[k][tumor_labels[k] == 2], bins=bins, range=xrng, density=True)
            # Convert float64 to float32
            slice_hists[k, :bins] = np.nan_to_num(val1.astype(np.float32))
            slice_hists[k, bins:] = np.nan_to_num(val2.astype(np.float32))
        np.save(dst_file % PID, slice_hists)
        print("{:03d} {:47s}".format(i, dst_file % PID))


def run_dump_hist_feature(num=-1):
    data_dir = Path(__file__).parent.parent.parent / "data/LiTS/Training_Batch"
    features_dir = Path(__file__).parent.parent.parent / "data/LiTS/feat/hist"
    dump_hist_feature(data_dir, features_dir, mode="train", bins=100,
                      xrng=(GRAY_MIN + 50, GRAY_MAX - 50), number=num)
    dump_hist_feature(data_dir, features_dir, mode="eval", bins=100,
                      xrng=(GRAY_MIN + 50, GRAY_MAX - 50), number=num)


def dump_hist_feature_v2(in_path, out_path,
                         mode="train",
                         bins=100,
                         xrng=(GRAY_MIN, GRAY_MAX),
                         number=0):
    """
    Compute histogram features for context guide

    Parameters
    ----------
    in_path: str
        data directory
    out_path: str
        npy/npz directory
    mode: str
        for train or eval
    bins: int
        number of bins of histogram, default 100
    xrng: tuple
        (x_min, x_max), default (GRAY_MIN, GRAY_MAX)
    number: int
        for debug

    Returns
    -------

    """
    src_path = Path(in_path)
    dst_path = Path(out_path) / mode
    dst_path.mkdir(parents=True, exist_ok=True)
    dst_file = str(dst_path / "%03d")

    for i, vol_case in enumerate(sorted(src_path.glob("volume-*.nii"),
                                        key=lambda x: int(str(x).split(".")[0].split("-")[-1]))):
        if number >= 0 and number != i:
            continue
        PID = int(vol_case.stem.split("-")[-1])

        vh, volume = nii_kits.read_lits(vol_case.stem.split("-")[-1], "vol", vol_case)
        lab_case = vol_case.parent / vol_case.name.replace("volume", "segmentation")
        _, labels = nii_kits.read_lits(vol_case.stem.split("-")[-1], "lab", lab_case)
        assert volume.shape == labels.shape, "Vol{} vs Lab{}".format(volume.shape, labels.shape)

        if mode == "train":
            gpl = [np.where(sli == 2) for sli in labels]
        else:
            gpl = array_kits.guide_pixel_list(
                labels, obj_val=2, guide="middle", tile_guide=True)
            print("gpl length in tumors slices: ", [len(x[0]) for x in gpl if len(x[0]) > 0])

        slice_hists = np.empty((volume.shape[0], bins * 2), dtype=np.float32)
        for k in range(volume.shape[0]):
            with np.errstate(invalid='ignore'):
                val1, _ = np.histogram(volume[k][labels[k] >= 1], bins=bins, range=xrng, density=True)
                val2, _ = np.histogram(volume[gpl[k][0], gpl[k][1], gpl[k][2]], bins=bins, range=xrng, density=True)
            # Convert float64 to float32
            slice_hists[k, :bins] = np.nan_to_num(val1.astype(np.float32))
            slice_hists[k, bins:] = np.nan_to_num(val2.astype(np.float32))
        np.save(dst_file % PID, slice_hists)
        print("{:03d} {:47s}".format(i, dst_file % PID))


def run_dump_hist_feature_v2(num=-1):
    data_dir = Path(__file__).parent.parent.parent / "data/LiTS/Training_Batch"
    features_dir = Path(__file__).parent.parent.parent / "data/LiTS/feat/hist"
    dump_hist_feature_v2(data_dir, features_dir, mode="eval", bins=100,
                         xrng=(GRAY_MIN + 50, GRAY_MAX - 50), number=num)


def dump_glcm_feature_for_train(in_path, out_path,
                                distances=(1, 2, 3),
                                angles=(0, 1, 2, 3),
                                level=256,
                                symmetric=True,
                                normed=True,
                                features="all",
                                filter_size=20,
                                average_num=1,
                                norm_levels=True,
                                number=-1):
    """
    Compute GLCM texture features for context guide

    Parameters
    ----------
    in_path: str
        data directory
    out_path: str
        npy/npz directory
    distances: tuple
    angles: tuple
    level: int
    symmetric: bool
    normed: bool
    features: str or list of str
        all for all the features or GLCM feature names
    filter_size: int
    average_num: int
        Collect at least 'average_num' feature vectors each for an image patch for
        getting average results
    norm_levels: bool
        Adjust GLCM feature scales to avoid extreme values
    number: int
        for debug
    """
    import warnings

    src_path = Path(in_path)
    dst_path = Path(out_path) / "train"
    dst_path.mkdir(parents=True, exist_ok=True)
    dst_file = str(dst_path / "%03d")
    with (Path(__file__).parent / "prepare/meta.json").open() as f:
        meta = json.load(f)
        meta = {case["PID"]: case for case in meta}
    angle = np.pi / 4

    mmax = {"contrast": -np.inf,
            "dissimilarity": -np.inf,
            "homogeneity": -np.inf,
            "energy": -np.inf,
            "entropy": -np.inf,
            "correlation": -np.inf,
            "cluster_shade": -np.inf,
            "cluster_prominence": -np.inf}
    mmin = {"contrast": np.inf,
            "dissimilarity": np.inf,
            "homogeneity": np.inf,
            "energy": np.inf,
            "entropy": np.inf,
            "correlation": np.inf,
            "cluster_shade": np.inf,
            "cluster_prominence": np.inf}
    for i, vol_case in enumerate(sorted(src_path.glob("volume-*.nii"),
                                        key=lambda x: int(str(x).split(".")[0].split("-")[-1]))):
        if number >= 0 and number != i:
            continue
        PID = int(vol_case.stem.split("-")[-1])
        print("{:03d} {:47s}".format(i, str(vol_case)))
        case = meta[PID]

        vh, volume = nii_kits.read_lits(vol_case.stem.split("-")[-1], "vol", vol_case)
        # lab_case = vol_case.parent / vol_case.name.replace("volume", "segmentation")
        # _, labels = nii_kits.read_lits(vol_case.stem.split("-")[-1], "lab", lab_case)
        # assert volume.shape == labels.shape, "Vol{} vs Lab{}".format(volume.shape, labels.shape)
        volume = (np.clip(volume, GRAY_MIN, GRAY_MAX) - GRAY_MIN) * (255. / (GRAY_MAX - GRAY_MIN))
        volume = volume.astype(np.uint8)

        if isinstance(features, str) and features != "all":
            feat_list = [features]
        elif isinstance(features, (list, tuple)):
            feat_list = features
        else:
            feat_list = ["contrast", "dissimilarity", "homogeneity", "energy", "entropy", "correlation",
                         "cluster_shade", "cluster_prominence"]
        glcm_feature_length = len(distances) * len(angles) * len(feat_list)
        slice_glcm_features = np.zeros((volume.shape[0], glcm_feature_length),
                                       dtype=np.float32)
        for k in range(volume.shape[0]):
            # Choice a series of Liver / Tumor patches
            if k in case["tumor_slices_index"]:
                ind = case["tumor_slices_index"].index(k)
                feat_vals = []
                counter = 0
                for j in range(case["tumor_slices_from_to"][ind], case["tumor_slices_from_to"][ind + 1]):
                    if case["tumor_slices_areas"][j] < filter_size:
                        continue
                    y1, x1, y2, x2 = case["tumor_slices"][j]
                    image_patch = volume[k, y1:y2, x1:x2]
                    image_patch = ndi.gaussian_filter(image_patch, 0.5)
                    _, ff = array_kits.glcm_features(image_patch, distances, [angle * x for x in angles],
                                                     level, symmetric, normed, feat_list,
                                                     flat=True, norm_levels=norm_levels)
                    feat_vals.append(np.array([ff[fe] for fe in feat_list]).reshape(-1))
                    for fe in feat_list:
                        mmax[fe] = max(mmax[fe], ff[fe].max())
                        mmin[fe] = min(mmin[fe], ff[fe].min())
                    counter += 1
                if counter == 0:
                    # All tumor areas in current slice are less than filter_size, so we omit this slice
                    continue
                if counter < average_num:
                    # Try to compute more glcm features for average results(more stable).
                    loop = 1
                    while True:
                        for j in range(case["tumor_slices_from_to"][ind], case["tumor_slices_from_to"][ind + 1]):
                            if case["tumor_slices_areas"][j] < filter_size:
                                continue
                            y1, x1, y2, x2 = case["tumor_slices"][j]
                            # Resize for new glcm features
                            with warnings.catch_warnings():
                                warnings.simplefilter("ignore")
                                image_patch = ndi.zoom(volume[k, y1:y2, x1:x2], (1. + loop * .1, ) * 2, order=1)
                            image_patch = ndi.gaussian_filter(image_patch, 0.5)
                            _, ff = array_kits.glcm_features(image_patch, distances, [angle * x for x in angles],
                                                             level, symmetric, normed, feat_list,
                                                             flat=True, norm_levels=norm_levels)
                            feat_vals.append(np.array([ff[fe] for fe in feat_list]).reshape(-1))
                            for fe in feat_list:
                                mmax[fe] = max(mmax[fe], ff[fe].max())
                                mmin[fe] = min(mmin[fe], ff[fe].min())
                            counter += 1
                        if counter >= average_num:
                            break
                        loop += 1
                slice_glcm_features[k] = np.mean(feat_vals, axis=0)
            # TODO(zjw): Liver glcm features, are we need?
        np.save(dst_file % PID, slice_glcm_features)
    print(mmax)
    print(mmin)


def dump_glcm_feature_for_eval(in_path, out_path,
                               distances=(1, 2, 3),
                               angles=(0, 1, 2, 3),
                               level=256,
                               symmetric=True,
                               normed=True,
                               features="all",
                               filter_size=20,
                               average_num=1,
                               norm_levels=True,
                               number=-1):
    """
    Compute GLCM texture features for context guide

    Parameters
    ----------
    in_path: str
        data directory
    out_path: str
        npy/npz directory
    distances: tuple
    angles: tuple
    level: int
    symmetric: bool
    normed: bool
    features: str or list of str
        all for all the features or GLCM feature names
    filter_size: int
    average_num: int
        Collect at least 'average_num' feature vectors each for an image patch for
        getting average results
    norm_levels: bool
        Adjust GLCM feature scales to avoid extreme values
    number: int
        for debug
    """
    import warnings

    src_path = Path(in_path)
    dst_path = Path(out_path) / "eval"
    dst_path.mkdir(parents=True, exist_ok=True)
    dst_file = str(dst_path / "%03d")
    with (Path(__file__).parent / "prepare/meta.json").open() as f:
        meta = json.load(f)
        meta = {case["PID"]: case for case in meta}
    angle = np.pi / 4

    mmax = {"contrast": -np.inf,
            "dissimilarity": -np.inf,
            "homogeneity": -np.inf,
            "energy": -np.inf,
            "entropy": -np.inf,
            "correlation": -np.inf,
            "cluster_shade": -np.inf,
            "cluster_prominence": -np.inf}
    mmin = {"contrast": np.inf,
            "dissimilarity": np.inf,
            "homogeneity": np.inf,
            "energy": np.inf,
            "entropy": np.inf,
            "correlation": np.inf,
            "cluster_shade": np.inf,
            "cluster_prominence": np.inf}
    for i, vol_case in enumerate(sorted(src_path.glob("volume-*.nii"),
                                        key=lambda x: int(str(x).split(".")[0].split("-")[-1]))):
        if number >= 0 and number != i:
            continue
        PID = int(vol_case.stem.split("-")[-1])
        print("{:03d} {:47s}".format(i, str(vol_case)))
        case = meta[PID]

        vh, volume = nii_kits.read_lits(vol_case.stem.split("-")[-1], "vol", vol_case)
        # lab_case = vol_case.parent / vol_case.name.replace("volume", "segmentation")
        # _, labels = nii_kits.read_lits(vol_case.stem.split("-")[-1], "lab", lab_case)
        # assert volume.shape == labels.shape, "Vol{} vs Lab{}".format(volume.shape, labels.shape)
        volume = (np.clip(volume, GRAY_MIN, GRAY_MAX) - GRAY_MIN) * (255. / (GRAY_MAX - GRAY_MIN))
        volume = volume.astype(np.uint8)

        if isinstance(features, str) and features != "all":
            feat_list = [features]
        elif isinstance(features, (list, tuple)):
            feat_list = features
        else:
            feat_list = ["contrast", "dissimilarity", "homogeneity", "energy", "entropy", "correlation",
                         "cluster_shade", "cluster_prominence"]
        glcm_feature_length = len(distances) * len(angles) * len(feat_list)
        slice_glcm_features = np.zeros((volume.shape[0], glcm_feature_length), dtype=np.float32)
        glcm_slice_counter = np.zeros((volume.shape[0],), dtype=np.int32)
        for tid in range(len(case["tumors"])):
            z1, y1, x1, z2, y2, x2 = case["tumors"][tid]
            middle_sid = (z2 - z1 - 1) // 2 + z1
            assert middle_sid in case["tumor_slices_index"]
            ind = case["tumor_slices_index"].index(middle_sid)
            feat_vals = []
            for j in range(case["tumor_slices_from_to"][ind], case["tumor_slices_from_to"][ind + 1]):
                if case["tumor_slices_tid"][j] == tid:
                    if case["tumor_slices_areas"][j] < filter_size:
                        break
                    y1, x1, y2, x2 = case["tumor_slices"][j]
                    image_patch = volume[middle_sid, y1:y2, x1:x2]
                    image_patch = ndi.gaussian_filter(image_patch, 0.5)
                    _, ff = array_kits.glcm_features(image_patch, distances, [angle * x for x in angles],
                                                     level, symmetric, normed, feat_list,
                                                     flat=True, norm_levels=norm_levels)
                    feat_vals.append(np.array([ff[fe] for fe in feat_list]).reshape(-1))
                    for fe in feat_list:
                        mmax[fe] = max(mmax[fe], ff[fe].max())
                        mmin[fe] = min(mmin[fe], ff[fe].min())
                    # Try to compute more glcm features for average results(more stable).
                    for loop in range(1, average_num):
                        # Resize for new glcm features
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            image_patch = ndi.zoom(volume[middle_sid, y1:y2, x1:x2], (1. + loop * .1, ) * 2, order=1)
                        image_patch = ndi.gaussian_filter(image_patch, 0.5)
                        _, ff = array_kits.glcm_features(image_patch, distances, [angle * x for x in angles],
                                                         level, symmetric, normed, feat_list,
                                                         flat=True, norm_levels=norm_levels)
                        feat_vals.append(np.array([ff[fe] for fe in feat_list]).reshape(-1))
                        for fe in feat_list:
                            mmax[fe] = max(mmax[fe], ff[fe].max())
                            mmin[fe] = min(mmin[fe], ff[fe].min())
                    slice_glcm_features[z1:z2] += np.tile(np.mean(feat_vals, axis=0, keepdims=True), (z2 - z1, 1))
                    glcm_slice_counter[z1:z2] += 1
                    break   # only use one slice per tumor
                    # TODO(zjw): Liver glcm features, are we need?
        glcm_slice_counter = np.clip(glcm_slice_counter, 1, np.inf)
        slice_glcm_features /= glcm_slice_counter[:, None]
        np.save(dst_file % PID, slice_glcm_features)
    print(mmax)
    print(mmin)


def run_dump_glcm_feature_for_train(norm_levels=True):
    data_dir = Path(__file__).parent.parent.parent / "data/LiTS/Training_Batch"
    features_dir = Path(__file__).parent.parent.parent / "data/LiTS/feat/glcm"
    dump_glcm_feature_for_train(data_dir, features_dir, average_num=3, norm_levels=norm_levels)


def run_dump_glcm_feature_for_eval(norm_levels=True):
    data_dir = Path(__file__).parent.parent.parent / "data/LiTS/Training_Batch"
    features_dir = Path(__file__).parent.parent.parent / "data/LiTS/feat/glcm"
    dump_glcm_feature_for_eval(data_dir, features_dir, average_num=3, norm_levels=norm_levels)


def simulate_user_prior(simul_name):
    """
    We assume user labels the middle slice of all the tumors by an ellipse, which can
    be represented by a center and a stddev. Meanwhile tumor position in z direction
    is also provided for better performance.
    """
    prepare_dir = Path(__file__).parent / "prepare"
    all_prior_dict = {}

    # Check existence
    obj_file = prepare_dir / simul_name
    if obj_file.exists():
        with obj_file.open() as f:
            dataset_dict = json.load(f)
        return dataset_dict

    with (Path(__file__).parent / "prepare/meta.json").open() as f:
        meta = json.load(f)
    for case in meta:
        case_dict = {}
        for tid in range(len(case["tumors"])):
            z1, y1, x1, z2, y2, x2 = case["tumors"][tid]
            middle_sid = (z2 - z1 - 1) // 2 + z1
            ind = case["tumor_slices_index"].index(middle_sid)
            for j in range(case["tumor_slices_from_to"][ind], case["tumor_slices_from_to"][ind + 1]):
                if case["tumor_slices_tid"][j] == tid:
                    obj_dict = {"z": [z1, z2],
                                "center": case["tumor_slices_centers"][j],
                                "stddev": case["tumor_slices_stddevs"][j]}
                    if middle_sid in case_dict:
                        case_dict[middle_sid].append(obj_dict)
                    else:
                        case_dict[middle_sid] = [obj_dict]
        all_prior_dict[case["PID"]] = case_dict

    # src_path = Path(in_path)
    # for i, lab_case in enumerate(sorted(src_path.glob("segmentation-*.nii"),
    #                                     key=lambda x: int(str(x).split(".")[0].split("-")[-1]))):
    #     print("{:03d} {:47s}".format(i, str(lab_case)))
    #     pid = int(lab_case.stem.split("-")[-1])
    #
    #     _, labels = nii_kits.read_lits(pid, "vol", lab_case)
    #     all_prior_dict[str(pid)] = array_kits.get_moments_multi_objs(
    #             labels, obj_value=2, partial=True, partial_slice="middle")
    #     for k, v_list in all_prior_dict[str(pid)].items():
    #         for j, v in enumerate(v_list):
    #             all_prior_dict[str(pid)][k][j]["stddev"] = [round(x, 3) for x in v["stddev"]]

    with obj_file.open("w") as f:
        json.dump(all_prior_dict, f)


def run_simulate_user_prior():
    simulate_user_prior("prior.json")


def test_set_label():
    data_dir = Path("E:/Dataset/LiTS/Test_Batch")
    lab_dir = Path("D:/Library/Downloads/ee")
    out_dir = Path(__file__).parent.parent.parent / "data/LiTS/Test_Label"
    out_dir.mkdir(parents=True, exist_ok=True)
    for test_file in data_dir.glob("test-volume-*.nii"):
        pid = int(test_file.name.split(".")[0].split("-")[-1])
        if pid == 59:
            continue
        print(pid)
        header = nii_kits.read_nii(test_file, only_header=True)
        spx, spy = abs(header["pixdim"][1]), abs(header["pixdim"][2])
        labels = np.zeros(header.get_data_shape()[::-1], dtype=np.uint8)
        for lab_file in lab_dir.glob("test-volume-{}-*.txt".format(pid)):
            sid = int(lab_file.name.split(".")[0].split("-")[3]) - 1
            points = np.loadtxt(str(lab_file)) / [spx, spy]
            points = points.astype(np.int32)
            cv2.fillPoly(labels[sid], [points], 1)
        out = nib.Nifti1Image(labels.transpose((2, 1, 0)), affine=None, header=header)
        nib.save(out, str(out_dir / "test-inter-{}.nii.gz".format(pid)))


def gen_infer_context(guide_file, test_meta_file,
                      bins=100,
                      xrng=(GRAY_MIN, GRAY_MAX)):
    guide_file = Path(guide_file)
    test_meta_file = Path(test_meta_file)
    with guide_file.open() as f:
        guide_list = json.load(f)
    with test_meta_file.open() as f:
        test_meta = json.load(f)
        test_meta = {x["PID"]: x for x in test_meta}

    hist_out_dir = Path(__file__).parent.parent.parent / "data/LiTS/feat/hist/infer"
    hist_out_dir.mkdir(parents=True, exist_ok=True)
    hist_dst_file = str(hist_out_dir / "%03d")
    # glcm_out_dir = Path(__file__).parent.parent.parent / "data/LiTS/feat/glcm/infer"
    # glcm_out_dir.mkdir(parents=True, exist_ok=True)
    for k in guide_list.keys():
        PID = int(k)
        case = test_meta[PID]
        guide = guide_list[k]
        d = case["size"][0]

        gpl = [[[], [], []] for _ in range(d)]
        for kk, vv in guide.items():
            sid = int(kk)
            x = np.arange(512)
            coords = np.stack(np.meshgrid(x, x, indexing="ij"), axis=-1)
            for t in vv:
                pi, pj = np.where(np.sum(((coords - t["center"]) / (np.array(t["stddev"]) / 0.7413)) ** 2, axis=-1) <= 1)
                for m in range(t["z"][0], t["z"][1]):
                    gpl[m][0].extend([sid] * len(pi))
                    gpl[m][1].extend(pi)
                    gpl[m][2].extend(pj)

        _, volume = nii_kits.read_nii(ROOT_DIR / case["vol_case"])
        _, labels = nii_kits.read_nii(ROOT_DIR / case["lab_case"])
        slice_hists = np.empty((volume.shape[0], bins * 2), dtype=np.float32)
        for n in range(volume.shape[0]):
            with np.errstate(invalid='ignore'):
                val1, _ = np.histogram(volume[n][labels[n] >= 1], bins=bins, range=xrng, density=True)
                val2, _ = np.histogram(volume[gpl[n][0], gpl[n][1], gpl[n][2]], bins=bins, range=xrng, density=True)
            # Convert float64 to float32
            slice_hists[n, :bins] = np.nan_to_num(val1.astype(np.float32))
            slice_hists[n, bins:] = np.nan_to_num(val2.astype(np.float32))
        np.save(hist_dst_file % PID, slice_hists)
        print("{:47s}".format(hist_dst_file % PID))


def run_gen_infer_context():
    guide_file = Path(__file__).parent / "prepare/interaction.json"
    test_meta_file = Path(__file__).parent / "prepare/test_meta_update.json"
    gen_infer_context(guide_file, test_meta_file, xrng=(GRAY_MIN + 50, GRAY_MAX - 50))


if __name__ == "__main__":
    cmd = input("Please choice function:\n\t"
                "a: exit()\n\t"
                "b: run_nii_3d_to_png()\n\t"
                "c: run_dump_hist_feature()\n\t"
                "c2: run_dump_hist_feature_v2()\n\t"
                "d: run_simulate_user_prior()\n\t"
                "e: run_dump_glcm_feature_for_train()\n\t"
                "f: run_dump_glcm_feature_for_eval()\n\t"
                "g: test_set_label()\n\t"
                "h: run_gen_infer_context() [A/b/c/...] ")
    cmd = cmd.lower()

    if cmd == "b":
        run_nii_3d_to_png()
    elif cmd == "c":
        run_dump_hist_feature()
    elif cmd == "c2":
        run_dump_hist_feature_v2()
    elif cmd == "d":
        run_simulate_user_prior()
    elif cmd == "e":
        norm_levels = input("Do you want to scale GLCM features according to levels? [Y/n]")
        if norm_levels in ["", "y", "Y"]:
            print("Enable feature scale.")
            run_dump_glcm_feature_for_train(True)
        else:
            run_dump_glcm_feature_for_train(False)
    elif cmd == "f":
        norm_levels = input("Do you want to scale GLCM features according to levels? [Y/n]")
        if norm_levels in ["", "y", "Y"]:
            print("Enable feature scale.")
            run_dump_glcm_feature_for_eval(True)
        else:
            run_dump_glcm_feature_for_eval(False)
    elif cmd == "g":
        test_set_label()
    elif cmd == "h":
        run_gen_infer_context()

    print("Exit")
