# Copyright 2019 Zhang Jianwei All Right Reserved.
#
# TODO: Choice a License
#
# =================================================================================

"""Converts LiTS liver data to TFRecord file format with Example protobuf.

3D CT image will be splitted into 2D slices.

LiTS liver dataset is expected to have the following directory structure:

  + MedicalImageSegmentation
    + data_kits
      - build_data.py
      - build_voc2012_data.py (current working directory).
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
  ./data/LiTS/origin

Semantic segmentation annotations:
  ./data/LiTS/mask

list folder:
  ./data/image_list.txt

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

