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
          - data_fold_0.tfrecord
          - data_fold_1.tfrecord
          - data_fold_2.tfrecord
          - data_fold_3.tfrecord
          - data_fold_4.tfrecord

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

from data_kits.build_data import ImageReader, image_to_examples
from data_kits.preprocess import random_split_k_fold
from utils.array_kits import find_empty_slices

LiTS_Dir = Path(__file__).parent.parent / "data" / "LiTS"


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
    assert dataset in ["trainval", "test"], "Wrong dataset: " + dataset
    image_files = []
    dataset = LiTS_Dir / "{}.txt".format(dataset)
    for line in dataset.open():
        parts = line.strip("\n").split(" ")
        if not keep_only_liver and len(parts) > 1 and parts[1] == "liver":
            continue
        image_files.append(parts[0])

    return sorted(image_files)


def read_k_folds(path, file_names):
    folds = []
    with Path(path).open() as f:
        for line in f.readlines():
            folds.append([int(x) for x in line[line.find(":") + 1:].strip().split(" ")])

    k_fold_names = [[] for _ in folds]
    all_nums = {int(x.replace(".nii", "").split("-")[-1]): x for x in file_names}
    for i, fold in enumerate(folds):
        for num in fold:
            if num in all_nums:
                k_fold_names[i].append(all_nums[num])

    return k_fold_names


def convert_data_set(dataset, keep_only_liver, k_split=5, seed=None):
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

    folds_list_save_to = Path(__file__).parent.parent / "data/LiTS/k_folds.txt"
    if folds_list_save_to.exists():
        k_folds = read_k_folds(folds_list_save_to, file_names)
    else:
        if k_split < 1:
            raise ValueError("k_split should >= 1, but get {}".format(k_split))
        k_folds = random_split_k_fold(file_names, k_split, seed) if k_split > 1 else [file_names]

        with folds_list_save_to.open("w") as f:
            for i, fold in enumerate(k_folds):
                f.write("Fold %d:" % i)
                name_nums = [int(name.replace(".nii", "").split("-")[-1]) for name in fold]
                write_str = " ".join([str(x) for x in sorted(name_nums)])
                f.write(write_str + "\n")

    image_reader = ImageReader(np.int16, extend_channel=True)
    label_reader = ImageReader(np.uint8, extend_channel=False)     # use uint8 to save space

    LiTS_records = LiTS_Dir / "records"
    if not LiTS_records.exists():
        LiTS_records.mkdir(parents=True, exist_ok=True)

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
                    extra_info = {"empty": find_empty_slices(label_reader.image())}
                    # Convert 2D slices to example
                    for example in image_to_examples(image_reader, label_reader, split=True,
                                                     extra_int_info=extra_info):
                        writer_2d.write(example.SerializeToString())
                        # Convert 3D volume to example
                    for example in image_to_examples(image_reader, label_reader, split=False):
                        writer_3d.write(example.SerializeToString())
                    counter += 1
                print()

