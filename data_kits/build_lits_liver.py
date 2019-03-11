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
      - build_lits_liver.py (current working directory).
    + data
      + LiTS
        - image_list.txt
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
  ./data/LiTS/Training_Batch_1
  ./data/LiTS/Training_Batch_2
  ./data/LiTS/Test_Batch

list folder:
  ./data/LiTS/trainval.txt
  ./data/LiTS/test.txt

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

import numpy as np
import tensorflow as tf
from pathlib import Path

from data_kits.build_data import ImageReader
from data_kits.build_data import SubVolumeReader
from data_kits.build_data import image_to_examples
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


def convert_to_liver(dataset, keep_only_liver, k_split=5, seed=None):
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

    k_folds = read_or_create_k_folds(Path(__file__).parent.parent / "data/LiTS/k_folds.txt",
                                     file_names, k_split, seed)
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
                                  min_bbox_shape=None):
    """
    Convert dataset list to several tf-record files.

    Strip boundary of the CT images.
    """
    file_names = get_lits_list(dataset, keep_only_liver)
    num_images = len(file_names)

    k_folds = read_or_create_k_folds(Path(__file__).parent.parent / "data/LiTS/k_folds.txt",
                                     file_names, k_split, seed)
    LiTS_records = _get_lits_records_dir()

    image_reader = SubVolumeReader(np.int16, extend_channel=True)
    label_reader = SubVolumeReader(np.uint8, extend_channel=False)  # use uint8 to save space

    # Convert each split
    counter = 1
    for i, fold in enumerate(k_folds):
        # Split to 2D slices for training
        output_filename_2d = LiTS_records / "{}-bbox-2D-{}-of-{}.tfrecord".format(dataset, i + 1, k_split)
        # 3D volume for evaluation and prediction
        output_filename_3d = LiTS_records / "{}-bbox-3D-{}-of-{}.tfrecord".format(dataset, i + 1, k_split)
        with tf.io.TFRecordWriter(str(output_filename_2d)) as writer_2d:
            with tf.io.TFRecordWriter(str(output_filename_3d)) as writer_3d:
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
                                only_tumor=False):
    file_names = get_lits_list(dataset, keep_only_liver if not only_tumor else False)
    num_images = len(file_names)

    k_folds = read_or_create_k_folds(Path(__file__).parent.parent / "data/LiTS/k_folds.txt",
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
                bbox = array_kits.extract_region(label_reader.image(), align, padding, min_bbox_shape)
                label_reader.bbox = bbox.tolist()
                image_reader.read(image_file)
                image_reader.bbox = bbox.tolist()

                if only_tumor:
                    # Extract tumor slices
                    tumor_value = np.max(label_reader.image())
                    indices = np.where(np.max(label_reader.image(), axis=(1, 2)) == tumor_value)[0]
                    image_reader.indices = indices
                    label_reader.indices = indices
                    print("  #Tumor slices: {:d}".format(len(indices)), end="", flush=True)
                    tumor_slices.extend(indices)

                # we have extended extra dimension for image
                if image_reader.shape[:-1] != label_reader.shape:
                    raise RuntimeError("Shape mismatched between image and label: {} vs {}".format(
                        image_reader.shape, label_reader.shape))
                # Convert 2D slices to example
                for example in image_to_examples(image_reader, label_reader, split=True, group=group):
                    writer_2d.write(example.SerializeToString())
                counter += 1
            print()

    if only_tumor:
        print("Total #tumor slices: {}".format(len(tumor_slices)))


def dump_fp_bbox_from_prediction(label_dirs, pred_dir):
    pred_dir = Path(pred_dir)
    save_path = pred_dir.parent / "fp_tp_bbox.pkl"
    # with save_path.open("rb") as f:
    cnt = 0
    for pred_path in pred_dir.glob("prediction-*.nii.gz"):
        print(pred_path.name)
        lab_file = pred_path.stem.replace("prediction", "segmentation")
        lab_path = misc.find_file(label_dirs, lab_file)
        result = array_kits.merge_labels(nii_kits.nii_reader(pred_path)[1], [0, 2])
        reference = array_kits.merge_labels(nii_kits.nii_reader(lab_path)[1], [0, 2])
        fps, tps = array_kits.find_tp_and_fp(result, reference)
        for x in fps:
            print(x, [x[3] - x[0], x[4] - x[1], x[5] - x[2]])
        print()
        cnt += 1


if __name__ == "__main__":
    dump_fp_bbox_from_prediction(
        label_dirs=[r"D:\DataSet\LiTS\Training_Batch_1",
                    r"D:\DataSet\LiTS\Training_Batch_2"],
        pred_dir=Path(__file__).parent.parent / "model_dir/004_triplet/prediction"
    )
