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
import argparse
import numpy as np
import tensorflow as tf
from pathlib import Path
import matplotlib
matplotlib.use("qt5Agg")
import matplotlib.pyplot as plt

import input_pipeline_nf
input_pipeline = input_pipeline_nf


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
    group.add_argument("--eval_skip_num",
                       type=int,
                       default=0,
                       required=False, help="Skip some cases for evaluating determined case")
    group.add_argument("--num_gpus",
                       type=int,
                       default=1,
                       required=False, help="Number of gpus to run this model")
    group.add_argument("--evaluator",
                       type=str,
                       choices=["Volume", "Slice", "NFVolume"])


class CheckInputPipeline(object):
    def setUp(self):
        record1 = Path(__file__).parent / "data/NF/records/trainval-2D-1-of-5.tfrecord"
        record1_3d = Path(__file__).parent / "data/NF/records/trainval-3D-1-of-5.tfrecord"
        self.records = [str(record1)]
        # self.records_3d = [str(record1_3d), str(record2_3d)]
        self.records_3d = [str(record1_3d)]

        self.parser = argparse.ArgumentParser()
        add_arguments(self.parser)
        input_pipeline.add_arguments(self.parser)
        sys.argv.extend([
            "--classes", "NF",
            "--im_height", "384",
            "--im_width", "288",
            "--resize_for_batch"
        ])
        self.sess = tf.Session()

    def test_get_multi_records_dataset_for_train(self):
        sys.argv.extend([
            "--mode", "train",
            "--filter_size", "0",
            "--input_group", "3",
        ])
        self.args = self.parser.parse_args()
        print(self.args)

        dataset = input_pipeline.get_2d_multi_records_dataset_for_train(self.records, self.args)
        inputs = dataset.make_one_shot_iterator().get_next("Inputs")
        cnt = 1000
        cnt2 = 0
        cnt3 = 0
        while cnt:
            if cnt % 100 == 0:
                print(cnt)
            features, labels = self.sess.run(inputs)
            cnt3 += np.max(labels, axis=(1, 2)).sum()
            # print(features["images"].shape)
            # print(features["names"][0])
            # print(features["id"])
            # print(labels.shape)
            # print(features["images"].max(), features["images"].min())
            # print(labels.max(), labels.min())
            # for i in range(8):
            #     plt.subplot(231)
            #     plt.imshow(features["images"][i, ..., 0], cmap="gray")
            #     plt.subplot(232)
            #     plt.imshow(features["images"][i, ..., 1], cmap="gray")
            #     plt.title("{}".format(cnt2))
            #     plt.subplot(233)
            #     plt.imshow(features["images"][i, ..., 2], cmap="gray")
            #     plt.subplot(234)
            #     plt.imshow(labels[i], cmap="gray")
            #     plt.show()
            #     cnt2 += 1
            #     plt.pause(3)
            cnt -= 1
        print(cnt3)

    def test_get_2d_multi_records_dataset_for_eval(self):
        sys.argv.extend([
            "--mode", "train",
            "--input_group", "3",
            "--filter_size", "0",
            "--batch_size", "1",
            "--evaluator", "NFVolume",
        ])
        self.args = self.parser.parse_args()
        print(self.args)

        dataset = input_pipeline.get_3d_multi_records_dataset_for_eval(self.records_3d, "eval_while_train", self.args)
        inputs = dataset.make_one_shot_iterator().get_next("Inputs")
        while True:
            features, labels = self.sess.run(inputs)
            print(features["images"].shape)
            # print(features["name"])
            # print(features["id"])
            print(labels.shape)
            # print(features["images"].max(), features["images"].min())
            # print(labels.max(), labels.min())
            plt.subplot(241)
            plt.imshow(features["images"][0, ..., 0], cmap="gray")
            plt.subplot(242)
            plt.imshow(features["images"][0, ..., 1], cmap="gray")
            plt.subplot(243)
            plt.imshow(features["images"][0, ..., 2], cmap="gray")
            plt.subplot(244)
            plt.imshow(labels[0], cmap="gray")
            plt.show()
            plt.pause(1)


if __name__ == "__main__":
    h = CheckInputPipeline()
    h.setUp()
    h.test_get_multi_records_dataset_for_train()
