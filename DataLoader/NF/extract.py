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
import multiprocessing
from pathlib import Path

import SimpleITK as sitk
import nibabel as nib
import numpy as np
import scipy.ndimage as ndi

from DataLoader.Liver import nii_kits
from utils import array_kits

GRAY_MIN = 0
GRAY_MAX = 1000
ROOT_DIR = Path(__file__).parent.parent.parent


def check_dataset(file_path):
    src_path = Path(file_path)
    print("Check dataset in %s" % src_path)
    for i, case in enumerate(sorted(src_path.glob("volume-*.nii.gz"),
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
    pid = int(vol_case.name.split(".")[0].split("-")[-1])

    vh, volume = nii_kits.read_nii(vol_case, out_dtype=np.int16)
    volume = np.clip(volume, GRAY_MIN, GRAY_MAX).astype(np.uint16)
    lab_case = vol_case.parent / vol_case.name.replace("volume", "segmentation")
    _, labels = nii_kits.read_nii(lab_case, out_dtype=np.uint8)
    labels = np.clip(labels, 0, 1)
    assert volume.shape == labels.shape, "Vol{} vs Lab{}".format(volume.shape, labels.shape)
    # print(vh)

    # 3D Tumor Information
    tumors, n_obj = ndi.label(labels == 1, disc3)
    slices = ndi.find_objects(tumors)
    objects = [[z.start, y.start, x.start, z.stop, y.stop, x.stop] for z, y, x in slices]
    all_centers, all_stddevs = [], []
    tumor_areas = []
    bbox_map_3dto2d = {i: {"centers": [], "stddevs": [], "areas": [], "slices": [], "z": []}
                       for i in range(len(slices))}
    z_rev_map = {i: {"tid": [], "rid": []} for i in range(volume.shape[0])}
    for j, sli in enumerate(slices):
        region = labels[sli] == 1
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

    meta_data = {"PID": pid,
                 "vol_case": str(vol_case),
                 "lab_case": str(lab_case),
                 "size": [int(x) for x in vh.get_data_shape()[::-1]],
                 "spacing": [float(x) for x in vh.get_zooms()[::-1]],
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
        for j, (img, lab) in enumerate(zip(volume, labels)):
            out_img_file = dst_dir / "{:03d}_im.png".format(j)
            out_img = sitk.GetImageFromArray(img)
            sitk.WriteImage(out_img, str(out_img_file))
            out_lab_file = dst_dir / "{:03d}_lb.png".format(j)
            out_lab = sitk.GetImageFromArray(lab)
            sitk.WriteImage(out_lab, str(out_lab_file))

    return meta_data


def nii_3d_to_png(in_path, out_path, only_meta=False):
    src_path = Path(in_path)
    dst_path = Path(out_path)
    json_file = dst_path / "meta.json"

    all_files = sorted(src_path.glob("volume-*.nii.gz"), key=lambda x: int(str(x).split(".")[0].split("-")[-1]))
    p = multiprocessing.Pool(4)
    all_meta_data = p.map(process_case,
                          zip(all_files, range(len(all_files)), [dst_path] * len(all_files),
                              [only_meta] * len(all_files)))
    all_meta_data = sorted(all_meta_data, key=lambda x: x["PID"])

    with json_file.open("w") as f:
        json.dump(all_meta_data, f)


def run_nii_3d_to_png():
    data_dir = Path(__file__).parent.parent.parent / "data/NF/nii_NF"
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
        png_dir = Path(__file__).parent.parent.parent / "data/NF/png"
        nii_3d_to_png(data_dir, png_dir, only_meta=only_meta)


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
    src_path = Path(in_path)
    dst_path = Path(out_path) / mode
    dst_path.mkdir(parents=True, exist_ok=True)
    dst_file = str(dst_path / "%03d")

    for i, vol_case in enumerate(sorted(src_path.glob("volume-*.nii.gz"),
                                        key=lambda x: int(str(x).split(".")[0].split("-")[-1]))):
        if number >= 0 and number != i:
            continue
        PID = int(vol_case.name.split(".")[0].split("-")[-1])
        print("{:03d} {:47s}".format(i, str(vol_case)))

        vh, volume = nii_kits.read_nii(vol_case)
        lab_case = vol_case.parent / vol_case.name.replace("volume", "segmentation")
        _, labels = nii_kits.read_nii(lab_case, np.uint8)
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
                val2, _ = np.histogram(volume[k][tumor_labels[k] == 1], bins=bins, range=xrng, density=True)
            # Convert float64 to float32
            slice_hists[k, :bins] = np.nan_to_num(val1.astype(np.float32))
            slice_hists[k, bins:] = np.nan_to_num(val2.astype(np.float32))
        np.save(dst_file % PID, slice_hists)


def run_dump_hist_feature(num=-1):
    data_dir = Path(__file__).parent.parent.parent / "data/NF/nii_NF"
    features_dir = Path(__file__).parent.parent.parent / "data/NF/feat/hist"
    dump_hist_feature(data_dir, features_dir, mode="train", bins=100,
                      xrng=(GRAY_MIN, GRAY_MAX), number=num)
    dump_hist_feature(data_dir, features_dir, mode="eval", bins=100,
                      xrng=(GRAY_MIN, GRAY_MAX), number=num)


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
    for i, vol_case in enumerate(sorted(src_path.glob("volume-*.nii.gz"),
                                        key=lambda x: int(str(x).split(".")[0].split("-")[-1]))):
        if number >= 0 and number != i:
            continue
        PID = int(vol_case.name.split(".")[0].split("-")[-1])
        print("{:03d} {:47s}".format(i, str(vol_case)))
        case = meta[PID]

        vh, volume = nii_kits.read_nii(vol_case)
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
    for i, vol_case in enumerate(sorted(src_path.glob("volume-*.nii.gz"),
                                        key=lambda x: int(str(x).split(".")[0].split("-")[-1]))):
        if number >= 0 and number != i:
            continue
        PID = int(vol_case.name.split(".")[0].split("-")[-1])
        print("{:03d} {:47s}".format(i, str(vol_case)))
        case = meta[PID]

        vh, volume = nii_kits.read_nii(vol_case)
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
    data_dir = Path(__file__).parent.parent.parent / "data/NF/nii_NF"
    features_dir = Path(__file__).parent.parent.parent / "data/NF/feat/glcm"
    dump_glcm_feature_for_train(data_dir, features_dir, average_num=3, norm_levels=norm_levels)


def run_dump_glcm_feature_for_eval(norm_levels=True):
    data_dir = Path(__file__).parent.parent.parent / "data/NF/nii_NF"
    features_dir = Path(__file__).parent.parent.parent / "data/NF/feat/glcm"
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

    with obj_file.open("w") as f:
        json.dump(all_prior_dict, f)


def run_simulate_user_prior():
    simulate_user_prior("prior.json")


if __name__ == "__main__":
    cmd = input("Please choice function:\n\t"
                "a: exit()\n\t"
                "b: run_nii_3d_to_png()\n\t"
                "c: run_dump_hist_feature()\n\t"
                "d: run_simulate_user_prior()\n\t"
                "e: run_dump_glcm_feature_for_train()\n\t"
                "f: run_dump_glcm_feature_for_eval() [A/b/c/d/e/f]")
    cmd = cmd.lower()

    if cmd == "b":
        run_nii_3d_to_png()
    elif cmd == "c":
        run_dump_hist_feature()
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

    print("Exit")
