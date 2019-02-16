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

import unittest
import argparse
import tensorflow as tf
from pathlib import Path
import matplotlib
matplotlib.use("qt5Agg")
import matplotlib.pyplot as plt

import input_pipeline


def add_arguments(parser):
    group = parser.add_argument_group(title="Input Pipeline Arguments")
    group.add_argument("--w_width",
                       type=float,
                       default=450,
                       required=False, help="Medical image window width (default: %(default)d)")
    group.add_argument("--w_level",
                       type=float,
                       default=50,
                       required=False, help="Medical image window level (default: %(default)d)")
    group.add_argument("--zoom",
                       action="store_true",
                       required=False, help="Augment dataset with random zoom in and shift")
    group.add_argument("--zoom_scale",
                       type=float,
                       default=1.5,
                       required=False, help="Maximum random zoom-in scale. Make sure zoom_scale >= 1. "
                                            "(default: %(default)f)")
    group.add_argument("--noise",
                       action="store_true",
                       required=False, help="Augment dataset with random noise")
    group.add_argument("--noise_scale",
                       type=float,
                       default=0.05,
                       required=False, help="Random noise scale (default: %(default)f)")
    group.add_argument("--batch_size",
                       type=int,
                       default=8,
                       required=False, help="Model batch size (default: %(default)d)")


class TestInputPipeline(unittest.TestCase):
    def setUp(self):
        record1 = Path(__file__).parent / "data/LiTS/records/test-2D-1-of-5.tfrecord"
        record2 = Path(__file__).parent / "data/LiTS/records/test-2D-2-of-5.tfrecord"
        record1_3d = Path(__file__).parent / "data/LiTS/records/test-3D-3-of-5.tfrecord"
        record2_3d = Path(__file__).parent / "data/LiTS/records/test-3D-2-of-5.tfrecord"
        self.records = [str(record1), str(record2)]
        self.records_3d = [str(record1_3d), str(record2_3d)]
        parser = argparse.ArgumentParser()
        add_arguments(parser)
        self.args = parser.parse_args()
        self.sess = tf.Session()

    def test_get_record_dataset_for_train(self):
        dataset = input_pipeline.get_record_dataset_for_train(self.records[0], self.args)
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

    def test_get_multi_records_dataset_for_train(self):
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

    def test_get_record_dataset_for_eval(self):
        with tf.name_scope("InputPipeline"):
            dataset = input_pipeline.get_record_dataset_for_eval(self.records_3d[0], self.args)
            inputs = dataset.make_one_shot_iterator().get_next("Inputs")
        # writer = tf.summary.FileWriter(logdir=str(Path(__file__).parent / "logs"), graph=self.sess.graph)
        # writer.close()
        while True:
            features, labels = self.sess.run(inputs)
            # print(len(features["images"]))
            # # print(features["name"])
            # # print(features["pad"])
            # print(labels.shape)
            # print(images.max(), images.min())
            # print(labels.max(), labels.min())
            print(features["names"][-1], features["pads"][-1], features["depth"][-1])
            # idx = int(features["images"].shape[0] * 0.75)
            idx = 6
            plt.subplot(221)
            plt.imshow(features["images"][idx, ..., 0], cmap="gray")
            plt.subplot(222)
            plt.imshow(features["images"][idx + 1, ..., 0], cmap="gray")
            plt.subplot(223)
            plt.imshow(labels[idx], cmap="gray")
            plt.subplot(224)
            plt.imshow(labels[idx + 1], cmap="gray")
            plt.show()
