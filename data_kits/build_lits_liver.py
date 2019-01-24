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
        + origin
        + mask
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

Semantic segmentation annotations:
  ./data/LiTS/mask

list folder:
  ./data/trainval.txt
  ./data/test.txt

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

_NUM_FOLDS = 5

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


def _convert_data_set(dataset, k_split=5, seed=None):
    file_names = get_lits_list(dataset)
    num_images = len(file_names)
    if k_split < 1:
        raise ValueError("k_split should >= 1, but get {}".format(k_split))
    k_folds = random_split_k_fold(file_names, k_split, seed) if k_split > 1 else [file_names]

    image_reader = ImageReader("nii", np.int16, extend_channel=True)
    label_reader = ImageReader("nii", np.uint8, extend_channel=False)     # use uint8 to save space

    # Convert each split
    for i, fold in enumerate(k_folds):
        output_filename = LiTS_Dir / "records" / "{}-{}-of-{}.tfrecord".format(dataset, i + 1, k_split)
        with tf.io.TFRecordWriter(output_filename) as writer:
            for j, image_name in enumerate(fold):
                print("\r>> Converting image {}/{} fold {}".format(j + 1, num_images, i + 1), flush=True)
                # Read image
                image_file = LiTS_Dir / image_name
                image_reader.read(image_file)
                seg_file = image_file.parent / image_file.name.replace("volume", "segmentation")
                label_reader.read(seg_file)
                if image_reader.shape[:-1] != label_reader.shape:   # we have extended extra dimension for image
                    raise RuntimeError("Shape mismatched between image and label: {} vs {}".format(
                        image_reader.shape, label_reader.shape))
                # Find empty slices(we skip them in training)
                extra_info = {"empty": find_empty_slices(label_reader.image())}
                # Convert to example
                for example in image_to_examples(image_reader, label_reader, extra_int_info=extra_info):
                    writer.write(example.SerializeToString())
        print(flush=True)
