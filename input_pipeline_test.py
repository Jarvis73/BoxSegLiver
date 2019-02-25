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

"""
Test for image_ops.py
"""

import sys
import unittest
import argparse
import tensorflow as tf
from pathlib import Path
import matplotlib
matplotlib.use("qt5Agg")
import matplotlib.pyplot as plt

import input_pipeline


def add_arguments(parser):
    group = parser.add_argument_group(title="Global Arguments")
    group.add_argument("--batch_size",
                       type=int,
                       default=8,
                       required=False, help="Model batch size (default: %(default)d)")
    group.add_argument("--classes",
                       type=str,
                       nargs="+",
                       required=True, help="Class names of the objects")
    group.add_argument("--mode",
                       type=str,
                       choices=["train", "eval", "infer"],
                       required=True, help="Model mode for train/val/test")


class TestInputPipeline(unittest.TestCase):
    def setUp(self):
        record1 = Path(__file__).parent / "data/LiTS/records/sample-bbox-2D-1-of-5.tfrecord"
        record2 = Path(__file__).parent / "data/LiTS/records/sample-bbox-2D-2-of-5.tfrecord"
        record1_3d = Path(__file__).parent / "data/LiTS/records/sample-bbox-3D-3-of-5.tfrecord"
        record2_3d = Path(__file__).parent / "data/LiTS/records/sample-bbox-3D-2-of-5.tfrecord"
        self.records = [str(record1), str(record2)]
        self.records_3d = [str(record1_3d), str(record2_3d)]

        self.parser = argparse.ArgumentParser()
        add_arguments(self.parser)
        input_pipeline.add_arguments(self.parser)
        sys.argv.remove(sys.argv[1])
        sys.argv.extend([
            "--classes", "Liver", "Tumor",
            "--im_height", "256",
            "--im_width", "256",
            "--resize_for_batch"
        ])
        self.sess = tf.Session()

    def test_get_multi_records_dataset_for_train(self):
        self.args = self.parser.parse_args()
        print(self.args)

        dataset = input_pipeline.get_multi_records_dataset_for_train(self.records, self.args)
        inputs = dataset.make_one_shot_iterator().get_next("Inputs")
        features, labels = self.sess.run(inputs)
        print(features["images"].shape)
        print(features["name"])
        print(features["id"])
        print(labels.shape)
        print(features["images"].max(), features["images"].min())
        print(labels.max(), labels.min())
        plt.subplot(221)
        plt.imshow(features["images"][0, ..., 0], cmap="gray")
        plt.subplot(222)
        plt.imshow(features["images"][1, ..., 0], cmap="gray")
        plt.subplot(223)
        plt.imshow(labels[0], cmap="gray")
        plt.subplot(224)
        plt.imshow(labels[1], cmap="gray")
        plt.show()


    def test_spatial_guide(self):
        sys.argv.extend([
            "--use_spatial_guide",
            "--mode", "train"
        ])
        self.args = self.parser.parse_args()
        print(self.args)

        dataset = input_pipeline.get_multi_records_dataset_for_train(self.records, self.args)
        # dataset = input_pipeline.get_multi_records_dataset_for_eval(self.records_3d, self.args)
        inputs = dataset.make_one_shot_iterator().get_next("Inputs")

        while True:
            features, labels = self.sess.run(inputs)
            print(features["images"].shape)
            # print(features["names"])
            # print(features["bboxes"])
            print(labels.shape)
            # print(features["images"].max(), features["images"].min())
            # print(labels.max(), labels.min())
            plt.subplot(221)
            plt.imshow(features["images"][0, ..., 0], cmap="gray")
            plt.subplot(222)
            plt.imshow(features["images"][0, ..., 1], cmap="gray")
            plt.subplot(223)
            plt.imshow(labels[0], cmap="gray")
            plt.show()
