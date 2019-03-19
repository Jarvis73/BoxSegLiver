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
import tensorflow as tf
from pathlib import Path
import matplotlib
matplotlib.use("qt5Agg")
import matplotlib.pyplot as plt
import numpy as np

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
    group.add_argument("--use_fewer_guide",
                       action="store_true",
                       required=False, help="Use fewer guide for evaluation")
    group.add_argument("--guide",
                       type=str,
                       default="first",
                       choices=["first", "middle"],
                       required=False, help="Generate guide from which slice")
    group.add_argument("--cls_branch",
                       action="store_true",
                       required=False, help="Classify branch")
    group.add_argument("--eval_skip_num",
                       type=int,
                       default=0,
                       required=False, help="Skip some cases for evaluating determined case")


class CheckInputPipeline(object):
    def setUp(self):
        record1 = Path(__file__).parent / "data/LiTS/records/sample-tumor-bb-triplet-2D-1-of-5.tfrecord"
        record2 = Path(__file__).parent / "data/LiTS/records/sample-tumor-bb-triplet-2D-2-of-5.tfrecord"
        record1_3d = Path(__file__).parent / "data/LiTS/records/sample-bbox-3D-3-of-5.tfrecord"
        record2_3d = Path(__file__).parent / "data/LiTS/records/sample-bbox-3D-2-of-5.tfrecord"
        self.records = [str(record1), str(record2)]
        self.records_3d = [str(record1_3d), str(record2_3d)]
        self.records_3d = [str(Path(__file__).parent / "data/LiTS/records/trainval-bbox-3D-1-of-5.tfrecord")]

        self.parser = argparse.ArgumentParser()
        add_arguments(self.parser)
        input_pipeline.add_arguments(self.parser)
        sys.argv.extend([
            "--classes", "Liver", "Tumor",
            "--im_height", "256",
            "--im_width", "256",
            "--resize_for_batch"
        ])
        self.sess = tf.Session()

    def test_get_multi_records_dataset_for_eval(self):
        sys.argv.extend([
            "--mode", "eval",
        ])
        self.args = self.parser.parse_args()
        print(self.args)

        dataset = input_pipeline.get_3d_multi_records_dataset_for_eval(self.records_3d, self.args)
        inputs = dataset.make_one_shot_iterator().get_next("Inputs")

        cnt = 0
        while True:
            features, labels = self.sess.run(inputs)
            print(features["images"].shape)
            print(features["bboxes"], flush=True)
            # print(features["name"])
            # print(features["id"])
            print(labels.shape, features["pads"])
            # print(features["images"].max(), features["images"].min())
            # print(labels.max(), labels.min())
            plt.subplot(121)
            plt.imshow(features["images"][0, ..., 0], cmap="gray")
            plt.subplot(122)
            plt.imshow(labels[0], cmap="gray")
            plt.show()
            cnt += 1
            if cnt > 20:
                break

    def test_get_multi_records_dataset_for_train(self):
        sys.argv.extend([
            "--mode", "train",
            "--only_tumor",
            "--filter_size", "10"
        ])
        self.args = self.parser.parse_args()
        print(self.args)

        dataset = input_pipeline.get_2d_multi_records_dataset_for_eval(self.records, self.args)
        inputs = dataset.make_one_shot_iterator().get_next("Inputs")
        while True:
            features, labels = self.sess.run(inputs)
            print(features["images"].shape)
            # print(features["name"])
            # print(features["id"])
            print(labels.shape)
            # print(features["images"].max(), features["images"].min())
            # print(labels.max(), labels.min())
            plt.subplot(231)
            plt.imshow(features["images"][0, ..., 0], cmap="gray")
            plt.subplot(232)
            plt.imshow(features["images"][0, ..., 1], cmap="gray")
            plt.subplot(233)
            plt.imshow(features["images"][0, ..., 2], cmap="gray")
            plt.subplot(234)
            plt.imshow(labels[0], cmap="gray")
            plt.subplot(235)
            plt.imshow(labels[1], cmap="gray")
            plt.subplot(236)
            plt.imshow(labels[2], cmap="gray")
            plt.show()
            plt.pause(3)

    def test_get_2d_multi_records_dataset_for_eval(self):
        sys.argv.extend([
            "--mode", "eval",
            "--only_tumor",
            "--input_group", "3",
            "--filter_size", "20",
            "--eval_2d"
        ])
        self.args = self.parser.parse_args()
        print(self.args)

        dataset = input_pipeline.get_2d_multi_records_dataset_for_eval([
            str(Path(__file__).parent / "data/LiTS/records/trainval-tumor-bbox-triplet-2D-1-of-5.tfrecord")
        ], self.args)
        inputs = dataset.make_one_shot_iterator().get_next("Inputs")
        while True:
            features, labels = self.sess.run(inputs)
            print(features["images"].shape)
            # print(features["name"])
            # print(features["id"])
            print(labels.shape)
            # print(features["images"].max(), features["images"].min())
            # print(labels.max(), labels.min())
            plt.subplot(231)
            plt.imshow(features["images"][0, ..., 0], cmap="gray")
            plt.subplot(232)
            plt.imshow(features["images"][0, ..., 1], cmap="gray")
            plt.subplot(233)
            plt.imshow(features["images"][0, ..., 2], cmap="gray")
            plt.subplot(234)
            plt.imshow(labels[0], cmap="gray")
            plt.subplot(235)
            plt.imshow(features["images"][1, ..., 1], cmap="gray")
            plt.subplot(236)
            plt.imshow(labels[1], cmap="gray")
            plt.show()
            plt.pause(3)

    def test_spatial_guide(self):
        sys.argv.extend([
            "--use_spatial_guide",
            "--mode", "train"
        ])
        self.args = self.parser.parse_args()
        print(self.args)

        dataset = input_pipeline.get_2d_multi_records_dataset_for_train(self.records, self.args)
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

    def test_get_image_boxes_dataset(self):
        sys.argv.extend([
            "--mode", "train",
            "--input_group", "3",
            "--only_tumor",
            "--filter_size", "20",
            "--cls_branch",
        ])
        self.args = self.parser.parse_args()
        self.args.batch_size = 1
        print(self.args)

        boxes = [
            str(Path(__file__).parent / "data/LiTS/records/cls-0tp-1-of-5.tfrecord"),
            str(Path(__file__).parent / "data/LiTS/records/cls-0tp-2-of-5.tfrecord")
        ]
        dataset = input_pipeline.get_images_and_bboxes_dataset(self.records, boxes, self.args)
        inputs = dataset.make_one_shot_iterator().get_next("Inputs")
        cnt = 0
        cnt2 = -1
        while True:
            features, labels = self.sess.run(inputs)
            print(features["images"].shape)
            print(features["names"][0])
            b = np.array(features["gt_boxes"][0][0])
            print(features["gt_boxes"])
            print(features["im_info"])
            print(labels.shape)
            # print(features["images"].max(), features["images"].min())
            # print(labels.max(), labels.min())
            if cnt2 < 0:
                cnt2 = features["bboxes"][0][2] + 1
            for i in range(self.args.batch_size):
                plt.subplot(231)
                plt.imshow(features["images"][i, ..., 0], cmap="gray")
                plt.subplot(232)
                plt.imshow(features["images"][i, ..., 1], cmap="gray")
                plt.title("{}".format(cnt2))
                plt.plot([b[1], b[3], b[3], b[1], b[1]], [b[0], b[0], b[2], b[2], b[0]])
                plt.subplot(233)
                plt.imshow(features["images"][i, ..., 2], cmap="gray")
                plt.subplot(234)
                plt.imshow(labels[i], cmap="gray")
                plt.plot([b[1], b[3], b[3], b[1], b[1]], [b[0], b[0], b[2], b[2], b[0]])
                plt.show()
                cnt2 += 1
                plt.pause(3)
                plt.close()
            # if cnt > 16:
            #     break
            cnt += 1

    def test_get_multi_channels_dataset_for_eval(self):
        sys.argv.extend([
            "--mode", "eval",
            "--input_group", "3",
            "--only_tumor",
        ])
        self.args = self.parser.parse_args()
        print(self.args)

        dataset = input_pipeline.get_3d_multi_channels_dataset_for_eval(self.records_3d, self.args)
        inputs = dataset.make_one_shot_iterator().get_next("Inputs")
        cnt = 0
        cnt2 = -1
        while True:
            features, labels = self.sess.run(inputs)
            print(features["images"].shape)
            # print(features["names"][0])
            # print(features["id"])
            print(labels.shape)
            # print(features["images"].max(), features["images"].min())
            # print(labels.max(), labels.min())
            if cnt2 < 0:
                cnt2 = features["bboxes"][0][2] + 1
            for i in range(8):
                plt.subplot(231)
                plt.imshow(features["images"][i, ..., 0], cmap="gray")
                plt.subplot(232)
                plt.imshow(features["images"][i, ..., 1], cmap="gray")
                plt.title("{}".format(cnt2))
                plt.subplot(233)
                plt.imshow(features["images"][i, ..., 2], cmap="gray")
                plt.subplot(234)
                plt.imshow(labels[i], cmap="gray")
                plt.show()
                cnt2 += 1
                plt.pause(1)
            cnt += 1


if __name__ == "__main__":
    h = CheckInputPipeline()
    h.setUp()
    h.test_get_multi_channels_dataset_for_eval()
