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

import math
import json
import random
import pandas as pd
import tensorflow as tf
from pathlib import Path
from tensorflow.python.platform import tf_logging as logging
from tensorflow.contrib import data as contrib_data

from utils import image_ops
from utils import distribution_utils

Dataset = tf.data.Dataset

# shuffle buffer size
SHUFFLE_BUFFER_SIZE = 1000


def add_arguments(parser):
    group = parser.add_argument_group(title="Input Pipeline Arguments")
    group.add_argument("--test_fold", type=int, default=0)
    group.add_argument("--im_height", type=int, default=384)
    group.add_argument("--im_width", type=int, default=288)
    group.add_argument("--im_channel", type=int, default=3)
    group.add_argument("--noise_scale", type=float, default=0.1)
    group.add_argument("--resize_for_batch", action="store_false")


def _get_datasets(test_fold, filter_size=5):
    NF_Dir = Path(__file__).parent / "data/NF"
    obj_file = NF_Dir / ("dataset_f%d_fs%d.json" % (test_fold, filter_size))
    if obj_file.exists():
        dataset_dict = json.load(obj_file.open())
        return dataset_dict

    datasets = NF_Dir / "k_folds.txt"
    with datasets.open() as f:
        k_folds = []
        for line in f.readlines():
            if line.strip():
                k_folds.append([int(x) for x in line[line.find(":") + 1:].strip().split(" ")])
    if test_fold + 1 > len(k_folds):
        raise ValueError("test_fold too large")
    testset = k_folds[test_fold]
    trainset = []
    for i, folds in enumerate(k_folds):
        if i != test_fold:
            trainset.extend(folds)

    info_file = NF_Dir / "nf_stat.csv"
    df = pd.read_csv(str(info_file))

    def parse(idx):
        parts = df[df["PID"] == idx][df["size/px"] > 5][["size/px", "x1", "y1", "z1",
                                                         "x2", "y2", "z2", "mean", "std",
                                                         "sz_x", "sz_y", "sz_z",
                                                         "sp_x", "sp_y", "sp_z"]]
        part_dict = parts.to_dict(orient="list")
        part_dict["PID"] = idx
        part_dict["mean"] = round(part_dict["mean"][0], 2)
        part_dict["std"] = round(part_dict["std"][0], 2)
        part_dict["sz_x"] = part_dict["sz_x"][0]
        part_dict["sz_y"] = part_dict["sz_y"][0]
        part_dict["sz_z"] = part_dict["sz_z"][0]
        part_dict["sp_x"] = part_dict["sp_x"][0]
        part_dict["sp_y"] = part_dict["sp_y"][0]
        part_dict["sp_z"] = part_dict["sp_z"][0]
        return part_dict

    dataset_dict = {"train": [], "test": []}
    for idx in sorted(trainset):
        dataset_dict["train"].append(parse(idx))
    for idx in sorted(testset):
        dataset_dict["test"].append(parse(idx))

    json.dump(dataset_dict, obj_file.open("w"))
    return dataset_dict


def _collect_datasets(test_fold, mode):
    dataset_dict = _get_datasets(test_fold)
    if mode == "train":
        return dataset_dict["train"]
    else:
        return dataset_dict["test"]


def input_fn(mode, params):
    if "args" not in params:
        raise KeyError("params of input_fn need an \"args\" key")

    args = params["args"]
    dataset = _collect_datasets(args.test_fold, mode)
    if len(dataset) == 0:
        raise ValueError("No valid dataset found!")

    with tf.variable_scope("InputPipeline"):
        if mode == "train":
            return get_dataset_for_train(dataset, args)
        elif mode in ["eval_while_train", "eval"]:
            return get_dataset_for_eval_while_train(dataset, args)


def data_processing_train(im1, im2, im3, seg_file, offset_x, offset_y, PID_si, mean, std, args):
    def parse(x):
        return tf.image.decode_png(tf.io.read_file(x), dtype=tf.uint16)
    img = tf.concat([parse(im1), parse(im2), parse(im3)], axis=-1)
    cropped_img = tf.image.crop_to_bounding_box(img, offset_y, offset_x, args.im_height, args.im_width)
    cropped_img = tf.cast(cropped_img, tf.float32)
    cropped_img = tf.clip_by_value((cropped_img - mean) / std, 0, 20)
    seg = tf.image.decode_png(tf.io.read_file(seg_file), dtype=tf.uint8)
    cropped_seg = tf.image.crop_to_bounding_box(seg, offset_y, offset_x, args.im_height, args.im_width)

    with tf.control_dependencies([tf.print(offset_x, offset_y)]):
        features = {"images": cropped_img, "names": PID_si}
        labels = tf.cast(tf.squeeze(cropped_seg, axis=-1), tf.int32)

    # features["images"] = image_ops.random_noise(features["images"], args.noise_scale)
    # logging.info("Train: Add random noise, scale = {}".format(args.noise_scale))
    # features["images"], labels = image_ops.random_flip_left_right(features["images"], labels)
    # logging.info("Train: Add random flip")

    return features, labels


def get_dataset_for_train(data_list, args):

    def data_gen():
        d = data_list
        max_val = len(d)
        max_len_nf = [len(x["x1"]) for x in d]
        vol_prefix = str(Path(__file__).parent / "data/NF/png_NF/volume-")
        seg_prefix = str(Path(__file__).parent / "data/NF/png_NF/segmentation-")
        target_h, target_w = args.im_height, args.im_width
        while True:
            bi = random.randint(0, max_val - 1)  # body index
            ni = random.randint(0, max_len_nf[bi] - 1)  # nf index
            si = random.randint(d[bi]["z1"][ni], d[bi]["z2"][ni] - 1)  # slice index
            PID = d[bi]["PID"]
            if si == 0:
                vols = [vol_prefix + "%03d_em.png" % PID,
                        vol_prefix + "%03d_%02d.png" % (PID, si),
                        vol_prefix + "%03d_%02d.png" % (PID, si + 1)]
            elif si == d[bi]["sz_z"] - 1:
                vols = [vol_prefix + "%03d_%02d.png" % (PID, si - 1),
                        vol_prefix + "%03d_%02d.png" % (PID, si),
                        vol_prefix + "%03d_em.png" % PID]
            else:
                vols = [vol_prefix + "%03d_%02d.png" % (PID, si - 1),
                        vol_prefix + "%03d_%02d.png" % (PID, si),
                        vol_prefix + "%03d_%02d.png" % (PID, si + 1)]
            seg = seg_prefix + "%03d_%02d.png" % (PID, si)
            cx = random.randint(d[bi]["x1"][ni], d[bi]["x2"][ni] - 1)
            cy = random.randint(d[bi]["y1"][ni], d[bi]["y2"][ni] - 1)
            offset_x = min(max(0, cx - target_w // 2 + random.randint(-25, 25)), d[bi]["sz_x"] - target_w)
            offset_y = min(max(0, cy - target_h // 2 + random.randint(-25, 25)), d[bi]["sz_y"] - target_h)
            yield vols[0], vols[1], vols[2], seg, offset_x, offset_y, ("%d_%d" % (PID, si)), \
                  d[bi]["mean"], d[bi]["std"]

    bs = distribution_utils.per_device_batch_size(args.batch_size, args.num_gpus)
    dataset = (tf.data.Dataset.from_generator(data_gen, (tf.string, tf.string, tf.string, tf.string,
                                                         tf.int32, tf.int32, tf.string, tf.float32, tf.float32))
               .shuffle(buffer_size=SHUFFLE_BUFFER_SIZE)
               .apply(tf.data.experimental.map_and_batch(lambda *args_: data_processing_train(*args_, args),
                                                         bs, num_parallel_batches=1))
               .prefetch(buffer_size=contrib_data.AUTOTUNE))

    return dataset


def data_processing_eval_while_train(im1, im2, im3, seg_file, offset_y, ori_width, PID_si, mean, std, args):
    def parse(x):
        return tf.image.decode_png(tf.io.read_file(x), dtype=tf.uint16)
    img = tf.concat([parse(im1), parse(im2), parse(im3)], axis=-1)
    cropped_img = tf.image.crop_to_bounding_box(img, offset_y, 0, args.im_height, ori_width)
    cropped_img = tf.cast(cropped_img, tf.float32)
    cropped_img = tf.clip_by_value((cropped_img - mean) / std, 0, 20)
    resized_img = tf.image.resize_bilinear(tf.expand_dims(cropped_img, axis=0),
                                           (args.im_height, args.im_width), align_corners=True)
    seg = tf.image.decode_png(tf.io.read_file(seg_file), dtype=tf.uint8)
    cropped_seg = tf.image.crop_to_bounding_box(seg, offset_y, 0, args.im_height, ori_width)
    resized_seg = tf.image.resize_nearest_neighbor(tf.expand_dims(cropped_seg, axis=0),
                                                   (args.im_height, args.im_width), align_corners=True)

    features = {"images": tf.squeeze(resized_img, axis=0), "names": PID_si}
    labels = tf.cast(tf.squeeze(tf.squeeze(resized_seg, axis=-1), axis=0), tf.int32)

    return features, labels


def get_dataset_for_eval_while_train(data_list, args):

    def data_gen():
        vol_prefix = str(Path(__file__).parent / "data/NF/png_NF/volume-")
        seg_prefix = str(Path(__file__).parent / "data/NF/png_NF/segmentation-")
        half_height = args.im_height // 2
        for case in data_list:
            PID = case["PID"]
            for si in range(case["sz_z"]):
                split_num = math.ceil(case["sz_y"] / half_height - 1)
                if si == 0:
                    vols = [vol_prefix + "%03d_em.png" % PID,
                            vol_prefix + "%03d_%02d.png" % (PID, si),
                            vol_prefix + "%03d_%02d.png" % (PID, si + 1)]
                elif si == case["sz_z"] - 1:
                    vols = [vol_prefix + "%03d_%02d.png" % (PID, si - 1),
                            vol_prefix + "%03d_%02d.png" % (PID, si),
                            vol_prefix + "%03d_em.png" % PID]
                else:
                    vols = [vol_prefix + "%03d_%02d.png" % (PID, si - 1),
                            vol_prefix + "%03d_%02d.png" % (PID, si),
                            vol_prefix + "%03d_%02d.png" % (PID, si + 1)]
                seg = seg_prefix + "%03d_%02d.png" % (PID, si)
                for patch_idx in range(split_num - 1):
                    yield vols[0], vols[1], vols[2], seg, \
                        patch_idx * half_height, case["sz_x"], ("%d_%d" % (PID, si)), case["mean"], case["std"]
                yield vols[0], vols[1], vols[2], seg, \
                    case["sz_y"] - half_height * 2, case["sz_x"], ("%d_%d" % (PID, si)), case["mean"], case["std"]

    bs = distribution_utils.per_device_batch_size(args.batch_size, args.num_gpus)
    dataset = (tf.data.Dataset.from_generator(data_gen, (tf.string, tf.string, tf.string, tf.string,
                                                         tf.int32, tf.int32, tf.string, tf.float32, tf.float32))
               .apply(tf.data.experimental.map_and_batch(lambda *args_: data_processing_eval_while_train(*args_, args),
                                                         bs, num_parallel_batches=1))
               .prefetch(buffer_size=contrib_data.AUTOTUNE))

    return dataset


if __name__ == "__main__":
    # _get_datasets("dataset", 0)
    # dd = _collect_datasets("dataset", 0, "train")
    import matplotlib.pyplot as plt
    random.seed(1234)
    tf.set_random_seed(1234)

    class Foo(object):
        test_fold = 0
        im_height = 384
        im_width = 288
        im_channel = 3
        noise_scale = 0.01
        resize_for_batch = True
        evaluator = "NFVolume"
        num_gpus = 1
        batch_size = 1


    params = {"args": Foo()}
    # ds = input_fn("eval_while_train", params).make_one_shot_iterator().get_next()
    ds = input_fn("train", params).make_one_shot_iterator().get_next()
    sess = tf.Session()

    def run(ds_):
        a, b = sess.run(ds_)
        print(a["names"][0])
        plt.subplot(141)
        plt.imshow(a["images"][0, :, :, 0], cmap="gray")
        plt.subplot(142)
        plt.imshow(a["images"][0, :, :, 1], cmap="gray")
        plt.subplot(143)
        plt.imshow(a["images"][0, :, :, 2], cmap="gray")
        plt.subplot(144)
        plt.imshow(b[0], cmap="gray")
        plt.show()
        # return a, b

    run(ds)

    # all_min, all_max = [], []
    # cnt = 0
    # while True:
    #     try:
    #         print("\rNumber:", cnt, end="")
    #         a, b = run(ds)
    #         all_min.append(a["images"].min())
    #         all_max.append(a["images"].max())
    #         cnt += 1
    #     except tf.errors.OutOfRangeError:
    #         break
    # print()
    # plt.subplot(121)
    # plt.hist(all_min)
    # plt.subplot(122)
    # plt.hist(all_max)
    # plt.show()

