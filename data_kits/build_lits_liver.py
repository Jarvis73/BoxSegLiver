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

"""Converts LiTS liver data to TFRecord file format with Example protobuf.

3D CT image will be splitted into 2D slices.

LiTS liver dataset is expected to have the following directory structure:

  + MedicalImageSegmentation
    + data_kits
      - build_data.py
      - build_lits_liver.py
    + data
      + LiTS
        - trainval.txt
        - k_fold.txt
        + records
          - trainval-2D-1-of-5.tfrecord
          - trainval-2D-2-of-5.tfrecord
          - trainval-2D-3-of-5.tfrecord
          - trainval-2D-4-of-5.tfrecord
          - trainval-2D-5-of-5.tfrecord
          - trainval-3D-1-of-5.tfrecord
          - trainval-3D-2-of-5.tfrecord
          - trainval-3D-3-of-5.tfrecord
          - trainval-3D-4-of-5.tfrecord
          - trainval-3D-5-of-5.tfrecord

Image folder:

  + MedicalImageSegmentation
    + data
      + NF
        + Training_Batch_1
          - volume-001.nii.gz
          - volume-002.nii.gz
          - segmentation-001.nii.gz
          - segmentation-002.nii.gz
        + Training_Batch_2
          - volume-041.nii.gz
          - volume-042.nii.gz
          - segmentation-041.nii.gz
          - segmentation-042.nii.gz
        + Test_Batch
          - volume-001.nii.gz
          - volume-002.nii.gz

This script converts data into shared data files and save at tfrecord folder.

The Example proto contains the following fields:

  image/encoded: encoded image content.
  image/filename: image filename.
  image/format: image file format.
  image/height: image height.
  image/width: image width.
  image/channels: image channels.
  image/segmentation/class/encoded: encoded semantic segmentation content.
  image/segmentation/class/format: semantic segmentation file format.
"""

import pickle
import numpy as np
import tensorflow as tf
from pathlib import Path

import data_kits.build_data as bd
from data_kits.build_data import ImageReader
from data_kits.build_data import SubVolumeReader
from data_kits.preprocess import random_split_k_fold
from utils import array_kits
from utils import misc
from utils import nii_kits

LiTS_Dir = Path(__file__).parent.parent / "data" / "LiTS"

IGNORE_TAG = ["liver", "outlier"]


def get_lits_list(dataset, keep_only_liver=True):
    """

    Parameters
    ----------
    dataset: str
        Choice dataset list: trainval.txt or test.txt
    keep_only_liver: bool
        Some cases only contain liver(which means no tumor).

    Returns
    -------

    """
    assert dataset in ["trainval", "test", "sample"], "Wrong dataset: " + dataset
    image_files = []
    dataset = LiTS_Dir / "{}.txt".format(dataset)
    for line in dataset.open():
        parts = line.strip("\n").split(" ")
        if not keep_only_liver and len(parts) > 1 and parts[1] in IGNORE_TAG:
            print("Ignore", parts[0], parts[1])
            continue
        image_files.append(parts[0])

    return sorted(image_files)


def read_or_create_k_folds(path, file_names, k_split=None, seed=None):
    path = Path(path)

    if path.exists():
        folds = []
        with path.open() as f:
            for line in f.readlines():
                folds.append([int(x) for x in line[line.find(":") + 1:].strip().split(" ")])

        k_folds = [[] for _ in folds]
        all_nums = {int(x.replace(".nii", "").split("-")[-1]): x for x in file_names}
        for i, fold in enumerate(folds):
            for num in fold:
                if num in all_nums:
                    k_folds[i].append(all_nums[num])

    else:
        if not isinstance(k_split, int) or k_split <= 0:
            raise ValueError("Wrong `k_split` value. Need a positive integer, got {}".format(k_split))
        if k_split < 1:
            raise ValueError("k_split should >= 1, but get {}".format(k_split))
        k_folds = random_split_k_fold(file_names, k_split, seed) if k_split > 1 else [file_names]

        with path.open("w") as f:
            for i, fold in enumerate(k_folds):
                f.write("Fold %d:" % i)
                name_nums = [int(name.replace(".nii", "").split("-")[-1]) for name in fold]
                write_str = " ".join([str(x) for x in sorted(name_nums)])
                f.write(write_str + "\n")

    for fold in k_folds:
        print([int(x.replace(".nii", "").split("-")[-1]) for x in fold])
    return k_folds


def _get_lits_records_dir():
    LiTS_records = LiTS_Dir / "records"
    if not LiTS_records.exists():
        LiTS_records.mkdir(parents=True, exist_ok=True)

    return LiTS_records


def image_to_examples(image_reader,
                      label_reader,
                      split=False,
                      extra_str_info=None,
                      extra_int_info=None,
                      group="triplet",
                      group_label=False,
                      mask_image=False,
                      extra_float_info=None):
    """
    Convert N-D image and label to N-D/(N-1)-D tfexamples.

    This is a generator.

    Parameters
    ----------
    image_reader: ImageReader
        Containing image
    label_reader: ImageReader
        Containing label
    split: bool
        If true, image and label will be converted slice by slice.
    extra_str_info: dict
        extra information added to tfexample. Dict keys will be the tf-feature keys,
        and values must be a list whose length is equal with image_reader.shape[0]
    extra_int_info: dict
    group: str
        none or triplet or quintuplet
    group_label: bool
    mask_image: bool
    extra_float_info: dict

    Returns
    -------
    A list of TF-Example

    """
    if group not in ["none", "triplet", "quintuplet"]:
        raise ValueError("group must be one of [none, triplet, quintuplet], got {}".format(group))

    if image_reader.name is None:
        raise RuntimeError("ImageReader need call `read()` first.")
    extra_str_split, extra_str_origin = bd._check_extra_info_type(extra_str_info)
    extra_int_split, extra_int_origin = bd._check_extra_info_type(extra_int_info)
    extra_float_split, extra_float_origin = bd._check_extra_info_type(extra_float_info)

    if split:
        num_slices = image_reader.shape[0]
        for extra_list in extra_str_split.values():
            assert num_slices == len(extra_list), "Length not equal: {} vs {}".format(num_slices, len(extra_list))
        for extra_list in extra_int_split.values():
            assert num_slices == len(extra_list), "Length not equal: {} vs {}".format(num_slices, len(extra_list))
        for extra_list in extra_float_split.values():
            assert num_slices == len(extra_list), "Length not equal: {} vs {}".format(num_slices, len(extra_list))

        for idx in image_reader.indices:
            if group == "triplet":
                indices = (idx - 1, idx, idx + 1)
                shape = image_reader.shape[1:-1] + (3,)
            elif group == "quintuplet":
                indices = (idx - 2, idx - 1, idx, idx + 1, idx + 2)
                shape = image_reader.shape[1:-1] + (5,)
            else:
                indices = idx
                shape = image_reader.shape[1:]
            if not mask_image:
                slices = image_reader.data(indices)
            else:
                slices = (image_reader.image(indices) * np.clip(label_reader.image(indices), 0, 1)
                          .astype(image_reader.image().dtype)).tobytes()
            feature_dict = {
                "image/encoded": bd._bytes_list_feature(slices),
                "image/name": bd._bytes_list_feature(image_reader.name),
                "image/format": bd._bytes_list_feature(image_reader.format),
                "image/shape": bd._int64_list_feature(shape),
                "segmentation/encoded": bd._bytes_list_feature(label_reader.data(indices if group_label else idx)),
                "segmentation/name": bd._bytes_list_feature(label_reader.name),
                "segmentation/format": bd._bytes_list_feature(label_reader.format),
                "segmentation/shape": bd._int64_list_feature(shape if group_label else label_reader.shape[1:]),
                "extra/number": bd._int64_list_feature(idx),
            }
            for key, val in extra_str_split.items():
                feature_dict["extra/{}".format(key)] = bd._bytes_list_feature(val[idx])
            for key, val in extra_int_split.items():
                feature_dict["extra/{}".format(key)] = bd._int64_list_feature(val[idx])
            for key, val in extra_float_split.items():
                feature_dict["extra/{}".format(key)] = bd._float_list_feature(val[idx])

            yield tf.train.Example(features=tf.train.Features(feature=feature_dict))
    else:
        feature_dict = {
            "image/encoded": bd._bytes_list_feature(image_reader.data()),
            "image/name": bd._bytes_list_feature(image_reader.name),
            "image/format": bd._bytes_list_feature(image_reader.format),
            "image/shape": bd._int64_list_feature(image_reader.shape),
            "segmentation/encoded": bd._bytes_list_feature(label_reader.data()),
            "segmentation/name": bd._bytes_list_feature(label_reader.name),
            "segmentation/format": bd._bytes_list_feature(label_reader.format),
            "segmentation/shape": bd._int64_list_feature(label_reader.shape)
        }

        for key, val in extra_str_origin.items():
            feature_dict["extra/{}".format(key)] = bd._bytes_list_feature(val)
        for key, val in extra_int_origin.items():
            feature_dict["extra/{}".format(key)] = bd._int64_list_feature(val)
        for key, val in extra_float_origin.items():
            feature_dict["extra/{}".format(key)] = bd._float_list_feature(val)
        yield tf.train.Example(features=tf.train.Features(feature=feature_dict))


def convert_to_liver(dataset, keep_only_liver, k_split=5, seed=None, folds_file="k_folds.txt"):
    """
    Convert dataset list to several tf-record files.

    Parameters
    ----------
    dataset: str
        Pass to `get_lits_list()`
    keep_only_liver: bool
        Pass to `get_lits_list()`
    k_split: int
        Number of splits of the tf-record files.
    seed: int
        Random seed to generate k-split lists.

    """
    file_names = get_lits_list(dataset, keep_only_liver)
    num_images = len(file_names)

    k_folds = read_or_create_k_folds(LiTS_Dir / folds_file, file_names, k_split, seed)
    LiTS_records = _get_lits_records_dir()

    image_reader = ImageReader(np.int16, extend_channel=True)
    label_reader = ImageReader(np.uint8, extend_channel=False)     # use uint8 to save space

    # Convert each split
    counter = 1
    for i, fold in enumerate(k_folds):
        # Split to 2D slices for training
        output_filename_2d = LiTS_records / "{}-2D-{}-of-{}.tfrecord".format(dataset, i + 1, k_split)
        # 3D volume for evaluation and prediction
        output_filename_3d = LiTS_records / "{}-3D-{}-of-{}.tfrecord".format(dataset, i + 1, k_split)
        with tf.io.TFRecordWriter(str(output_filename_2d)) as writer_2d:
            with tf.io.TFRecordWriter(str(output_filename_3d)) as writer_3d:
                for j, image_name in enumerate(fold):
                    print("\r>> Converting fold {}, {}/{}, {}/{}"
                          .format(i + 1, j + 1, len(fold), counter, num_images), end="")
                    # Read image
                    image_file = LiTS_Dir / image_name
                    image_reader.read(image_file)
                    seg_file = image_file.parent / image_file.name.replace("volume", "segmentation")
                    label_reader.read(seg_file)
                    # we have extended extra dimension for image
                    if image_reader.shape[:-1] != label_reader.shape:
                        raise RuntimeError("Shape mismatched between image and label: {} vs {}".format(
                            image_reader.shape, label_reader.shape))
                    # Find empty slices(we skip them in training)
                    extra_info = {"empty_split": array_kits.find_empty_slices(label_reader.image())}
                    # Convert 2D slices to example
                    for example in image_to_examples(image_reader, label_reader, split=True,
                                                     extra_int_info=extra_info):
                        writer_2d.write(example.SerializeToString())
                    # Convert 3D volume to example
                    for example in image_to_examples(image_reader, label_reader, split=False):
                        writer_3d.write(example.SerializeToString())
                    counter += 1
                print()


def convert_to_liver_bounding_box(dataset,
                                  keep_only_liver,
                                  k_split=5,
                                  seed=None,
                                  align=1,
                                  padding=0,
                                  min_bbox_shape=None,
                                  prefix="bbox",
                                  folds_file="k_folds.txt"):
    """
    Convert dataset list to several tf-record files.

    Strip boundary of the CT images.
    """
    file_names = get_lits_list(dataset, keep_only_liver)
    num_images = len(file_names)

    k_folds = read_or_create_k_folds(Path(__file__).parent.parent / "data/LiTS/{}".format(folds_file),
                                     file_names, k_split, seed)
    LiTS_records = _get_lits_records_dir()

    image_reader = SubVolumeReader(np.int16, extend_channel=True)
    label_reader = SubVolumeReader(np.uint8, extend_channel=False)  # use uint8 to save space

    # Convert each split
    counter = 1
    for i, fold in enumerate(k_folds):
        # Split to 2D slices for training
        output_filename_2d = LiTS_records / "{}-{}-2D-{}-of-{}.tfrecord".format(dataset, prefix, i + 1, k_split)
        # 3D volume for evaluation and prediction
        output_filename_3d = LiTS_records / "{}-{}-3D-{}-of-{}.tfrecord".format(dataset, prefix, i + 1, k_split)
        with tf.io.TFRecordWriter(str(output_filename_2d)) as writer_2d, \
                tf.io.TFRecordWriter(str(output_filename_3d)) as writer_3d:
            for j, image_name in enumerate(fold):
                print("\r>> Converting fold {}, {}/{}, {}/{}"
                      .format(i + 1, j + 1, len(fold), counter, num_images), end="")

                # Read image
                image_file = LiTS_Dir / image_name
                seg_file = image_file.parent / image_file.name.replace("volume", "segmentation")

                label_reader.read(seg_file)
                bbox = array_kits.extract_region(label_reader.image(), align, padding, min_bbox_shape)
                label_reader.bbox = bbox.tolist()
                image_reader.read(image_file)
                image_reader.bbox = bbox.tolist()
                print(" {}".format(image_reader.shape), end="")
                if not np.all(np.array(image_reader.shape)[:3] % align[::-1] == 0):
                    raise ValueError("{}: box {} shape {}".format(image_file.stem, bbox, image_reader.shape))

                # we have extended extra dimension for image
                if image_reader.shape[:-1] != label_reader.shape:
                    raise RuntimeError("Shape mismatched between image and label: {} vs {}".format(
                        image_reader.shape, label_reader.shape))
                #
                extra_info = {"bbox_origin": bbox}
                # Convert 2D slices to example
                for example in image_to_examples(image_reader, label_reader, split=True):
                    writer_2d.write(example.SerializeToString())
                    # Convert 3D volume to example
                for example in image_to_examples(image_reader, label_reader, split=False,
                                                 extra_int_info=extra_info):
                    writer_3d.write(example.SerializeToString())
                counter += 1
            print()


def convert_to_liver_bbox_group(dataset,
                                keep_only_liver,
                                k_split=5,
                                seed=None,
                                align=1,
                                padding=0,
                                min_bbox_shape=None,
                                prefix="bbox-none",
                                group="none",
                                only_tumor=False,
                                mask_image=False,
                                folds_file="k_folds.txt"):
    file_names = get_lits_list(dataset, keep_only_liver if not only_tumor else False)
    num_images = len(file_names)

    k_folds = read_or_create_k_folds(Path(__file__).parent.parent / "data/LiTS/{}".format(folds_file),
                                     file_names, k_split, seed)
    LiTS_records = _get_lits_records_dir()

    image_reader = SubVolumeReader(np.int16, extend_channel=True)
    label_reader = SubVolumeReader(np.uint8, extend_channel=False)  # use uint8 to save space

    tumor_slices = []
    # Convert each split
    counter = 1
    print("Group: {}".format(group))
    for i, fold in enumerate(k_folds):
        # Split to 2D slices for training
        output_filename_2d = LiTS_records / "{}-{}-2D-{}-of-{}.tfrecord".format(dataset, prefix, i + 1, k_split)
        print("Write to {}".format(str(output_filename_2d)))
        with tf.io.TFRecordWriter(str(output_filename_2d)) as writer_2d:
            for j, image_name in enumerate(fold):
                print("\r>> Converting fold {}, {}/{}, {}/{}"
                      .format(i + 1, j + 1, len(fold), counter, num_images), end="")

                # Read image
                image_file = LiTS_Dir / image_name
                seg_file = image_file.parent / image_file.name.replace("volume", "segmentation")

                label_reader.read(seg_file)
                bbox = array_kits.extract_region(np.squeeze(label_reader.image()), align, padding, min_bbox_shape)
                label_reader.bbox = bbox.tolist()
                image_reader.read(image_file)
                image_reader.bbox = bbox.tolist()
                print(" {}".format(image_reader.shape), end="")
                if not np.all(np.array(image_reader.shape)[:3] % align[::-1] == 0):
                    raise ValueError("{}: box {} shape {}".format(image_file.stem, bbox, image_reader.shape))

                if only_tumor:
                    # Extract tumor slices
                    tumor_value = np.max(label_reader.image())
                    indices = np.where(np.max(label_reader.image(), axis=(1, 2)) == tumor_value)[0]
                    image_reader.indices = indices
                    label_reader.indices = indices
                    print("  #Tumor slices: {:d}   ".format(len(indices)), end="", flush=True)
                    tumor_slices.extend(indices)

                # we have extended extra dimension for image
                if image_reader.shape[:-1] != label_reader.shape:
                    raise RuntimeError("Shape mismatched between image and label: {} vs {}".format(
                        image_reader.shape, label_reader.shape))
                # Convert 2D slices to example
                for example in image_to_examples(image_reader, label_reader, split=True,
                                                 group=group, group_label=only_tumor,
                                                 mask_image=mask_image):
                    writer_2d.write(example.SerializeToString())
                counter += 1
            print()

    if only_tumor:
        print("Total #tumor slices: {}".format(len(tumor_slices)))


# Deprecated
def dump_fp_bbox_from_prediction(label_dirs, pred_dir):
    pred_dir = Path(pred_dir)
    save_path = pred_dir.parent / "bboxes-{}.pkl".format(pred_dir.parent.name)

    all_bboxes = {}
    counter = 0
    for pred_path in sorted(pred_dir.glob("prediction-*.nii.gz")):
        print(pred_path.name)
        lab_file = pred_path.stem.replace("prediction", "segmentation")
        lab_path = misc.find_file(label_dirs, lab_file)
        result = array_kits.merge_labels(nii_kits.nii_reader(pred_path)[1], [0, 2])
        reference = array_kits.merge_labels(nii_kits.nii_reader(lab_path)[1], [0, 2])
        fps, tps = array_kits.find_tp_and_fp(result, reference)
        for x in fps:
            print(x, [x[3] - x[0], x[4] - x[1], x[5] - x[2]])
        print()
        counter += len(fps)
        for x in tps:
            print(x, [x[3] - x[0], x[4] - x[1], x[5] - x[2]])
        print("#" * 80)
        all_bboxes[int(pred_path.stem.replace(".nii", "").split("-")[-1])] = {
            "fps": fps, "tps": tps
        }
    print("FPs: {}".format(counter))

    with save_path.open("wb") as f:
        pickle.dump(all_bboxes, f, pickle.HIGHEST_PROTOCOL)


# Deprecated
def bbox_to_examples(label_reader, bbox_array):
    # bbox: shape [depth, 4] --> [y1, x1, y2, x2]
    # where point (y1, x1) and (y2, z2) is in region
    shape = label_reader.shape[1:]

    def norm_bbox(bboxes):
        bbox = bboxes.copy()
        bbox += np.array([[-3, -3, 3, 3]])
        bbox = np.clip(bbox, 0, 65535)
        bbox[:, 2:4] = np.minimum(bbox[:, 2:4], np.array(shape)[None, :])
        return bbox / (np.concatenate((shape, shape), axis=0) - 1)

    for idx in label_reader.indices:
        locs = norm_bbox(np.asarray(bbox_array[idx]).reshape(-1, 4))
        feature_dict = {
            "name": bd._bytes_list_feature(label_reader.name),
            "data": bd._bytes_list_feature(locs.astype(np.float32).tobytes()),
            "shape": bd._int64_list_feature(locs.shape)
        }

        yield tf.train.Example(features=tf.train.Features(feature=feature_dict))


# Deprecated
def convert_to_tp_dataset(dataset,
                          k_split=5,
                          folds_file="k_folds.txt",
                          seed=None,
                          align=1,
                          padding=0,
                          min_bbox_shape=None,
                          prefix="cls-0tp"):
    file_names = get_lits_list(dataset, False)
    num_images = len(file_names)

    k_folds = read_or_create_k_folds(Path(__file__).parent.parent / "data/LiTS/{}".format(folds_file),
                                     file_names, k_split, seed)
    LiTS_records = _get_lits_records_dir()

    label_reader = SubVolumeReader(np.uint8, extend_channel=False)

    counter = 1
    for i, fold in enumerate(k_folds):
        output_filename = LiTS_records / "{}-{}-of-{}.tfrecord".format(prefix, i + 1, k_split)
        with tf.io.TFRecordWriter(str(output_filename)) as writer:
            for j, image_name in enumerate(fold):
                image_file = LiTS_Dir / image_name
                print("\r>> Converting fold {}, {}/{}, {}/{}"
                      .format(i + 1, j + 1, len(fold), counter, num_images), end="")

                seg_file = image_file.parent / image_file.name.replace("volume", "segmentation")
                label_reader.read(seg_file)
                bbox = array_kits.extract_region(label_reader.image(), align, padding, min_bbox_shape)
                label_reader.bbox = bbox.tolist()

                tumor_value = np.max(label_reader.image())
                indices = np.where(np.max(label_reader.image(), axis=(1, 2)) == tumor_value)[0]
                label_reader.indices = indices

                tps = array_kits.find_tp(array_kits.merge_labels(label_reader.image(), [0, 2]),
                                         split=True)

                # list of fps is sorted by z, y, x
                for example in bbox_to_examples(label_reader, tps):
                    writer.write(example.SerializeToString())
                counter += 1
            print()


# Deprecated
def convert_to_classify_dataset(dataset,
                                pred_dir,
                                prefix,
                                k_split=5,
                                seed=None,
                                align=1,
                                padding=0,
                                min_bbox_shape=None):
    pred_dir = Path(pred_dir)
    file_names = get_lits_list(dataset, False)
    num_images = len(file_names)

    k_folds = read_or_create_k_folds(Path(__file__).parent.parent / "data/LiTS/k_folds.txt",
                                     file_names, k_split, seed)
    LiTS_records = _get_lits_records_dir()

    label_reader = SubVolumeReader(np.uint8, extend_channel=False)  # use uint8 to save space
    pred_reader = SubVolumeReader(np.uint8, extend_channel=False)

    counter = 1
    for i, fold in enumerate(k_folds):
        output_filename = LiTS_records / "cls-{}-{}-of-{}.tfrecord".format(prefix, i + 1, k_split)
        with tf.io.TFRecordWriter(str(output_filename)) as writer:
            for j, image_name in enumerate(fold):
                image_file = LiTS_Dir / image_name
                print("\r>> Converting fold {}, {}/{}, {}/{}"
                      .format(i + 1, j + 1, len(fold), counter, num_images), end="")

                seg_file = image_file.parent / image_file.name.replace("volume", "segmentation")
                label_reader.read(seg_file)
                bbox = array_kits.extract_region(label_reader.image(), align, padding, min_bbox_shape)
                label_reader.bbox = bbox.tolist()

                pred_file = pred_dir / image_file.name.replace("volume", "prediction")
                pred_file = pred_file.with_suffix(".nii.gz")
                pred_reader.read(pred_file)
                pred_reader.bbox = bbox.tolist()

                tumor_value = np.max(label_reader.image())
                indices = np.where(np.max(label_reader.image(), axis=(1, 2)) == tumor_value)[0]
                label_reader.indices = indices

                fps, _ = array_kits.find_tp_and_fp(pred_reader.image(), label_reader.image())
                # list of fps is sorted by z, y, x
                for example in bbox_to_examples(label_reader, fps):
                    writer.write(example.SerializeToString())
                counter += 1
            print()


def hist_to_examples(label_reader, liver_hist_array, tumor_hist_array, split=False):
    assert len(liver_hist_array.shape) == 2
    assert len(tumor_hist_array.shape) == 2
    if split:
        liver_length = liver_hist_array.shape[1]
        tumor_length = tumor_hist_array.shape[1]
        for idx in label_reader.indices:
            feature_dict = {
                "liver_hist/encoded": bd._bytes_list_feature(liver_hist_array[idx].tobytes()),
                "liver_hist/shape": bd._int64_list_feature(liver_length),
                "tumor_hist/encoded": bd._bytes_list_feature(tumor_hist_array[idx].tobytes()),
                "tumor_hist/shape": bd._int64_list_feature(tumor_length),
                "case/name": bd._bytes_list_feature(label_reader.name)
            }
            yield tf.train.Example(features=tf.train.Features(feature=feature_dict))
    else:
        feature_dict = {
            "liver_hist/encoded": bd._bytes_list_feature(liver_hist_array.tobytes()),
            "liver_hist/shape": bd._int64_list_feature(liver_hist_array.shape),
            "tumor_hist/encoded": bd._bytes_list_feature(tumor_hist_array.tobytes()),
            "tumor_hist/shape": bd._int64_list_feature(tumor_hist_array.shape),
            "case/name": bd._bytes_list_feature(label_reader.name)
        }
        yield tf.train.Example(features=tf.train.Features(feature=feature_dict))


def convert_to_histogram_dataset_deprecated(dataset,
                                            keep_only_liver,
                                            k_split=5,
                                            seed=None,
                                            prefix="hist",
                                            folds_file="k_folds.txt",
                                            bins=100,
                                            xrng=(-200, 250)):
    file_names = get_lits_list(dataset, keep_only_liver)
    num_images = len(file_names)

    k_folds = read_or_create_k_folds(Path(__file__).parent.parent / "data/LiTS/{}".format(folds_file),
                                     file_names, k_split, seed)
    LiTS_records = _get_lits_records_dir()

    image_reader = SubVolumeReader(np.int16, extend_channel=True)
    label_reader = SubVolumeReader(np.uint8, extend_channel=False)  # use uint8 to save space

    # Convert each split
    counter = 1
    for i, fold in enumerate(k_folds):
        output_filename_2d = LiTS_records / "{}-{}-{}_{}-2D-{}-of-{}.tfrecord"\
            .format(prefix, bins, xrng[0], xrng[1], i + 1, k_split)
        output_filename_3d = LiTS_records / "{}-{}-{}_{}-3D-{}-of-{}.tfrecord"\
            .format(prefix, bins, xrng[0], xrng[1], i + 1, k_split)
        print("Write to {} and {}".format(str(output_filename_2d), str(output_filename_3d)))
        with tf.io.TFRecordWriter(str(output_filename_2d)) as writer_2d, \
                tf.io.TFRecordWriter(str(output_filename_3d)) as writer_3d:
            for j, image_name in enumerate(fold):
                print("\r>> Converting fold {}, {}/{}, {}/{}"
                      .format(i + 1, j + 1, len(fold), counter, num_images), end="")

                # Read image
                image_file = LiTS_Dir / image_name
                seg_file = image_file.parent / image_file.name.replace("volume", "segmentation")

                label_reader.read(seg_file)
                image_reader.read(image_file)

                # we have extended extra dimension for image
                if image_reader.shape[:-1] != label_reader.shape:
                    raise RuntimeError("Shape mismatched between image and label: {} vs {}".format(
                        image_reader.shape, label_reader.shape))

                images = image_reader.image()
                labels = label_reader.image()
                liver, tumor = images[labels == 1], images[labels == 2]
                val1, _ = np.histogram(liver.flat, bins=bins, range=xrng, density=True)
                val2, _ = np.histogram(tumor.flat, bins=bins, range=xrng, density=True)
                # Convert float64 to float32
                val1_total = np.tile(val1[None, :].astype(np.float32), [len(label_reader.indices), 1])
                val2_total = np.tile(val2[None, :].astype(np.float32), [len(label_reader.indices), 1])

                # Convert 2D slices to example
                for example in hist_to_examples(label_reader, val1_total, val2_total, split=True):
                    writer_2d.write(example.SerializeToString())
                # Convert 3D slices to example
                for example in hist_to_examples(label_reader, val1_total, val2_total, split=False):
                    writer_3d.write(example.SerializeToString())
                counter += 1
            print()


def convert_to_histogram_dataset(dataset,
                                 keep_only_liver,
                                 k_split=5,
                                 seed=None,
                                 align=1,
                                 padding=0,
                                 min_bbox_shape=None,
                                 prefix="hist",
                                 folds_file="k_folds.txt",
                                 bins=100,
                                 xrng=(-200, 250),
                                 guide=None,
                                 hist_scale="total"):
    np.warnings.filterwarnings('ignore')
    if guide is not None and guide not in ["first", "middle"]:
        raise ValueError("`guide` must be None or chosen from [first, middle]")
    if hist_scale not in ["total", "slice"]:
        raise ValueError("`hist_scale` must be chosen from [total, slice]")

    file_names = get_lits_list(dataset, keep_only_liver)
    num_images = len(file_names)

    k_folds = read_or_create_k_folds(Path(__file__).parent.parent / "data/LiTS/{}".format(folds_file),
                                     file_names, k_split, seed)
    LiTS_records = _get_lits_records_dir()

    image_reader = SubVolumeReader(np.int16, extend_channel=False)
    label_reader = SubVolumeReader(np.uint8, extend_channel=False)  # use uint8 to save space

    # Convert each split
    counter = 1
    for i, fold in enumerate(k_folds):
        output_filename_2d = LiTS_records / "{}-{}-{}_{}-2D-{}-of-{}.tfrecord"\
            .format(prefix, bins, xrng[0], xrng[1], i + 1, k_split)
        output_filename_3d = LiTS_records / "{}-{}-{}_{}-3D-{}-of-{}.tfrecord"\
            .format(prefix, bins, xrng[0], xrng[1], i + 1, k_split)
        output_filename_2d_guide = LiTS_records / "{}-guide-{}-{}_{}-2D-{}-of-{}.tfrecord" \
            .format(prefix, bins, xrng[0], xrng[1], i + 1, k_split)
        output_filename_3d_guide = LiTS_records / "{}-guide-{}-{}_{}-3D-{}-of-{}.tfrecord" \
            .format(prefix, bins, xrng[0], xrng[1], i + 1, k_split)
        print("Write to {} and {}".format(str(output_filename_2d), str(output_filename_3d)))
        with tf.io.TFRecordWriter(str(output_filename_2d)) as writer_2d, \
                tf.io.TFRecordWriter(str(output_filename_3d)) as writer_3d, \
                tf.io.TFRecordWriter(str(output_filename_2d_guide)) as writer_2d_guide, \
                tf.io.TFRecordWriter(str(output_filename_3d_guide)) as writer_3d_guide:
            for j, image_name in enumerate(fold):
                print("\r>> Converting fold {}, {}/{}, {}/{}"
                      .format(i + 1, j + 1, len(fold), counter, num_images), end="")

                # Read image
                image_file = LiTS_Dir / image_name
                seg_file = image_file.parent / image_file.name.replace("volume", "segmentation")

                label_reader.read(seg_file)
                bbox = array_kits.extract_region(np.squeeze(label_reader.image()), align, padding, min_bbox_shape)
                label_reader.bbox = bbox.tolist()
                image_reader.read(image_file)
                image_reader.bbox = bbox.tolist()
                print(" {}".format(image_reader.shape), end="")
                if not np.all(np.array(image_reader.shape)[:3] % align[::-1] == 0):
                    raise ValueError("{}: box {} shape {}".format(image_file.stem, bbox, image_reader.shape))

                if image_reader.shape != label_reader.shape:
                    raise RuntimeError("Shape mismatched between image and label: {} vs {}".format(
                        image_reader.shape, label_reader.shape))

                images = image_reader.image()
                labels = label_reader.image()
                if hist_scale == "total":
                    liver = images[labels >= 1]
                    tumor = images[labels == 2]
                    val1, _ = np.histogram(liver, bins=bins, range=xrng, density=True)
                    val2, _ = np.histogram(tumor, bins=bins, range=xrng, density=True)
                    # Convert float64 to float32
                    val1_total = np.tile(val1[None, :].astype(np.float32), [len(label_reader.indices), 1])
                    val2_total = np.tile(val2[None, :].astype(np.float32), [len(label_reader.indices), 1])

                    if guide is not None:
                        guide_image = array_kits.get_guide_image(
                            labels, obj_val=2, guide=guide, tile_guide=True)
                        tumor_guide = images[guide_image == 1]
                        val3, _ = np.histogram(tumor_guide, bins=bins, range=xrng, density=True)
                        # Convert float64 to float32
                        val3_total = np.tile(val3[None, :].astype(np.float32), [len(label_reader.indices), 1])
                else:   # "slice"
                    liver_slice_hists, tumor_slice_hists = [], []
                    for k in range(images.shape[0]):
                        val1, _ = np.histogram(images[k][labels[k] >= 1], bins=bins, range=xrng, density=True)
                        val2, _ = np.histogram(images[k][labels[k] == 2], bins=bins, range=xrng, density=True)
                        # Convert float64 to float32
                        liver_slice_hists.append(val1.astype(np.float32))
                        tumor_slice_hists.append(val2.astype(np.float32))
                    val1_total = np.nan_to_num(np.array(liver_slice_hists))
                    val2_total = np.nan_to_num(np.array(tumor_slice_hists))

                    tumor_guide_slice_hists = []
                    if guide is not None:
                        tumor_labels = array_kits.get_guide_image(
                            labels, obj_val=2, guide=guide, tile_guide=True) * 2
                        for k in range(images.shape[0]):
                            val3, _ = np.histogram(images[k][tumor_labels[k] >= 1],
                                                   bins=bins, range=xrng, density=True)
                            # Convert float64 to float32
                            tumor_guide_slice_hists.append(val3.astype(np.float32))
                        val3_total = np.nan_to_num(np.array(tumor_guide_slice_hists))

                # Convert 2D slices to example
                for example in hist_to_examples(label_reader, val1_total, val2_total, split=True):
                    writer_2d.write(example.SerializeToString())
                # Convert 3D slices to example
                for example in hist_to_examples(label_reader, val1_total, val2_total, split=False):
                    writer_3d.write(example.SerializeToString())
                if guide is not None:
                    for example in hist_to_examples(label_reader, val1_total, val3_total, split=True):
                        writer_2d_guide.write(example.SerializeToString())
                    for example in hist_to_examples(label_reader, val1_total, val3_total, split=False):
                        writer_3d_guide.write(example.SerializeToString())

                counter += 1
            print()
