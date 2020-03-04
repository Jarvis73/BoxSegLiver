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


import cv2
import tqdm
import math
import json
import copy
import shutil
import random
import itertools
import numpy as np
import tensorflow as tf
from pathlib import Path

import tensorflow_estimator as tfes
from tensorflow.python.platform import tf_logging as logging
from tensorflow.contrib import data as contrib_data
from tensorflow.python.util import deprecation

from utils import image_ops
from utils import distribution_utils
from DataLoader import misc
from DataLoader.Liver import nii_kits

deprecation._PRINT_DEPRECATION_WARNINGS = False
ModeKeys = tfes.estimator.ModeKeys
Dataset = tf.data.Dataset
PROJ_ROOT = Path(__file__).parent.parent.parent
pattern = str(PROJ_ROOT / "data/LiTS/png/volume-{:d}/{:03d}_im.png")
GRAY_MIN = -200
GRAY_MAX = 250
IM_SCALE = 64
LB_SCALE = 64
RND_SCALE = (1.2, 1.5)


def add_arguments(parser):
    group = parser.add_argument_group(title="Input Pipeline Arguments")
    group.add_argument("--test_fold", type=int, default=2)
    group.add_argument("--im_height", type=int, default=32)
    group.add_argument("--im_width", type=int, default=32)
    group.add_argument("--im_channel", type=int, default=1)
    group.add_argument("--filter_size", type=int, default=0, help="Filter tumors small than the given size")
    group.add_argument("--noise_scale", type=float, default=0.01)
    group.add_argument("--zoom_scale", type=float, nargs=2, default=RND_SCALE)
    group.add_argument("--random_flip", type=int, default=1,
                       help="Random flip while training. 0 for no flip, 1 for flip only left/right, "
                            "2 for only up/down, 3 for left/right and up/down")
    group.add_argument("--eval_num_batches_per_epoch", type=int, default=100)
    group.add_argument("--eval_mirror", action="store_true")


def _get_datasets(test_fold=-1, filter_size=10, choices=None, exclude=None):
    prepare_dir = Path(__file__).parent / "prepare"
    prepare_dir.mkdir(parents=True, exist_ok=True)

    # Check existence
    obj_file = prepare_dir / ("dataset_f%d_fs%d.json" % (test_fold, filter_size))
    if obj_file.exists():
        dataset_dict = json.load(obj_file.open())
        return dataset_dict

    # Load meta.json
    meta_file = prepare_dir / "meta.json"
    if not meta_file.exists():
        src_meta = Path(__file__).parent.parent.parent / "data/NF/png/meta.json"
        if not src_meta.exists():
            raise FileNotFoundError(str(src_meta))
        shutil.copyfile(str(src_meta), str(meta_file))
    with meta_file.open() as f:
        meta = json.load(f)
    meta = {x["PID"]: x for x in meta}

    def parse(case):
        tumors = case.pop("tumors")
        tumor_areas = case.pop("tumor_areas")
        case.pop("tumor_centers")
        case.pop("tumor_stddevs")
        case.pop("tumor_slices_from_to")
        case.pop("tumor_slices_centers")
        case.pop("tumor_slices_stddevs")
        case.pop("tumor_slices_areas")
        case.pop("tumor_slices")
        new_tumors = []
        new_areas = []
        for i in range(len(tumors)):
            if tumor_areas[i] >= filter_size:
                new_tumors.append(tumors[i])
                new_areas.append(tumor_areas[i])
        case["tumors"] = new_tumors
        case["tumor_areas"] = new_areas
        case["num_tumors"] = len(new_tumors)
        return case

    if not choices:
        # Load k_folds
        fold_path = prepare_dir / "k_folds.txt"
        all_cases = list(meta)
        if exclude:
            for exc in exclude:
                all_cases.remove(exc)
        print("Read:: k folds, test fold = %d" % test_fold)
        k_folds = misc.read_or_create_k_folds(fold_path, all_cases, k_split=5, seed=1357)
        if test_fold + 1 > len(k_folds):
            raise ValueError("test_fold too large")
        if test_fold < 0:
            raise ValueError("test_fold must be non-negative")
        testset = k_folds[test_fold]
        trainset = []
        for i, folds in enumerate(k_folds):
            if i != test_fold:
                trainset.extend(folds)

        dataset_dict = {"train": [], "val": []}
        for idx in sorted([int(x) for x in trainset]):
            dataset_dict["train"].append(parse(meta[idx]))
        for idx in sorted([int(x) for x in testset]):
            dataset_dict["val"].append(parse(meta[idx]))

        with obj_file.open("w") as f:
            json.dump(dataset_dict, f)
    else:
        dataset_dict = {"choices": []}
        for idx in choices:
            dataset_dict["choices"].append(parse(meta[idx]))

    return dataset_dict


def _collect_datasets(test_fold, mode, filter_tumor_size=0):
    dataset_dict = _get_datasets(test_fold, filter_size=filter_tumor_size) #if mode != "infer" else _get_test_data()
    if mode == "train":
        return dataset_dict["train"]
    elif mode == "infer":
        return dataset_dict["infer"]
    else:
        return dataset_dict["val"]


def input_fn(mode, params):
    if "args" not in params:
        raise KeyError("params of input_fn need an \"args\" key")

    args = params["args"]
    dataset = _collect_datasets(args.test_fold, mode,
                                filter_tumor_size=args.filter_size)
    if len(dataset) == 0:
        raise ValueError("No valid dataset found!")

    with tf.variable_scope("InputPipeline"):
        if mode == ModeKeys.TRAIN:
            return get_dataset_for_train(dataset, random_scale=args.zoom_scale, config=args)
        elif mode == "eval_online":
            return get_dataset_for_eval_online(dataset, config=args)
        elif mode == ModeKeys.EVAL:
            return get_dataset_for_eval_image(dataset, config=args)
        elif mode == ModeKeys.PREDICT:
            raise NotImplementedError


#####################################
#
#   Input pipeline
#
#####################################

def get_dataset_for_train(data_list, random_scale=(1., 1.), config=None):
    batch_size = distribution_utils.per_device_batch_size(config.batch_size, config.num_gpus)

    def train_gen():
        return gen_train_batch(data_list, batch_size, random_scale, random_window_level=True, config=config)

    dataset = (tf.data.Dataset.from_generator(train_gen,
                                              (tf.string, tf.int32, tf.int32, tf.int32, tf.float32),
                                              output_shapes=(tf.TensorShape([2]),
                                                             tf.TensorShape([]),
                                                             tf.TensorShape([4]),
                                                             tf.TensorShape([]),
                                                             tf.TensorShape([2])))
               .apply(tf.data.experimental.map_and_batch(
                        lambda *args_: data_processing_train(*args_,
                                                             config=config,
                                                             random_noise=True,
                                                             random_flip_left_right=config.random_flip & 1 > 0,
                                                             random_flip_up_down=config.random_flip & 2 > 0),
                        batch_size, num_parallel_batches=1))
               .prefetch(buffer_size=contrib_data.AUTOTUNE))

    return dataset


def gen_train_batch(data_list,
                    batch_size,
                    random_scale=(1., 1.),
                    random_window_level=False,
                    config=None,
                    mode="Train"):
    _ = config
    d = data_list
    list_of_keys = np.arange(len(d))
    log_dataset(data_list, prefix=mode + " ")

    while True:
        ci = np.random.choice(list_of_keys, batch_size, True)   # case indices

        force_positive = batch_size // 2
        positive_count = 0
        for j, i in enumerate(ci):
            case = d[i]
            size = case["size"]
            pid = case["PID"]

            # Get selected tumor
            tid = random.randint(0, case["num_tumors"] - 1)
            tumor = case["tumors"][tid]

            # Get selected slice
            if tumor[3] - tumor[0] < 3:
                positive = 0
                sid = random.choice([tumor[0], tumor[3] - 1])
                if sid == 0:
                    sid += 1
                elif sid == size[0] - 1:
                    sid -= 1
            else:
                if positive_count < force_positive:
                    sid = random.randint(1, tumor[3] - 2) + tumor[0]
                    positive = 1
                    positive_count += 1
                else:
                    sid = random.choice([tumor[0], tumor[3] - 1])
                    if sid == 0 or sid == size[0] - 1:
                        sid = random.randint(1, tumor[3] - 2) + tumor[0]
                        positive = 1
                        positive_count += 1
                    else:
                        positive = 0
            im_files = [pattern.format(pid, sid - 1), pattern.format(pid, sid + 1)]

            y1, x1 = tumor[1:3]
            y2, x2 = tumor[4:6]
            crop_size = (np.array([y2 - y1, x2 - x1], np.float32) *
                         np.random.uniform(*random_scale, size=2)).astype(np.int32).tolist()
            cy, cx = (y2 + y1 - 1) // 2, (x2 + x1 - 1) // 2
            off_y, off_x = cy - crop_size[0] // 2, cx - crop_size[1] // 2

            # Random clip image value
            if random_window_level:
                img_clip = (random.randint(10, 50) * IM_SCALE * 1., random.randint(500, 540) * IM_SCALE * 1.)
            else:
                img_clip = (50 * IM_SCALE * 1., 500 * IM_SCALE * 1.)

            yield im_files, positive, [off_y, off_x] + crop_size, pid, img_clip


def data_processing_train(im_files, label, bbox, PID_ci, img_clip, config,
                          random_noise, random_flip_left_right, random_flip_up_down):
    off_x, off_y, height, width = bbox[0], bbox[1], bbox[2], bbox[3]

    im_file1, im_file2 = tf.split(im_files, 2)
    img1 = tf.image.decode_png(tf.io.read_file(im_file1), channels=1, dtype=tf.uint16)
    img2 = tf.image.decode_png(tf.io.read_file(im_file1), channels=1, dtype=tf.uint16)
    img = tf.stack([img1, img2], axis=0)
    img = tf.image.crop_to_bounding_box(img, off_x, off_y, height, width)
    img = tf.image.resize_bilinear(img, (config.im_height, config.im_width), align_corners=True)
    img = tf.cast(img, tf.float32)
    clip_min, clip_max = tf.split(img_clip, 2)
    img = (tf.clip_by_value(img, clip_min, clip_max) - clip_min) / (clip_max - clip_min)

    if random_noise:
        img = image_ops.random_noise(img, config.noise_scale)
        logging.info("Train: Add random noise, scale = {}".format(config.noise_scale))
    if random_flip_left_right:
        img = image_ops.random_flip_left_right(img)
        logging.info("Train: Add random flip left <-> right")
    if random_flip_up_down:
        img = image_ops.random_flip_up_down(img)
        logging.info("Train: Add random flip up <-> down")

    im1, im2 = tf.unstack(img, 2)
    features = {"im1": im1, "im2": im2, "names": PID_ci}
    labels = label

    return features, labels


def get_dataset_for_eval_online(data_list, config=None):
    batch_size = distribution_utils.per_device_batch_size(config.batch_size, config.num_gpus)

    def val_gen():
        infinity_generator = gen_train_batch(data_list, batch_size, random_scale=(RND_SCALE,) * 2, config=config,
                                             mode="Val")
        for _ in tqdm.tqdm(range(config.eval_num_batches_per_epoch * config.batch_size)):
            yield next(infinity_generator)

    dataset = (tf.data.Dataset.from_generator(val_gen, (tf.string, tf.int32, tf.int32, tf.int32, tf.float32),
                                              output_shapes=(tf.TensorShape([2]),
                                                             tf.TensorShape([]),
                                                             tf.TensorShape([4]),
                                                             tf.TensorShape([]),
                                                             tf.TensorShape([2])))
               .apply(tf.data.experimental.map_and_batch(
                        lambda *args_: data_processing_train(*args_,
                                                             config=config,
                                                             random_noise=False,
                                                             random_flip_left_right=False,
                                                             random_flip_up_down=False),
                        batch_size, num_parallel_batches=1))
               .prefetch(buffer_size=contrib_data.AUTOTUNE))

    return dataset


def get_dataset_for_eval_image(data_list, config=None):
    batch_size = config.batch_size
    eval_batch = {"im1": np.empty((batch_size, config.im_height, config.im_width, 1)),
                  "im2": np.empty((batch_size, config.im_height, config.im_width, 1)),
                  "mirror": 0,
                  "names": None}
    labels = [0] * batch_size
    bs = 0
    log_dataset(data_list, "Eval ")

    for ci, case in enumerate(data_list[config.eval_skip_num:config.eval_skip_num + config.eval_num]):
        size = case["size"]
        pid = case["PID"]
        eval_batch["names"] = pid
        for tumor in case["tumors"]:
            for sid in range(tumor[0], tumor[3]):
                if tumor[3] - tumor[0] < 3:
                    if sid == 0:
                        sid += 1
                    if sid == size[0] - 1:
                        sid -= 1
                    labels[bs] = 0
                elif sid == 0 or sid == size[0] - 1:
                    continue
                elif sid == tumor[0] or sid == tumor[3] - 1:
                    labels[bs] = 0
                else:
                    labels[bs] = 1

                y1, x1 = tumor[1:3]
                y2, x2 = tumor[4:6]
                crop_size = (np.array([y2 - y1, x2 - x1], np.float32) * ([RND_SCALE] * 2)).astype(np.int32).tolist()
                cy, cx = (y2 + y1 - 1) // 2, (x2 + x1 - 1) // 2
                off_y, off_x = cy - crop_size[0] // 2, cx - crop_size[1] // 2

                im1 = cv2.imread(pattern.format(pid, sid - 1), cv2.IMREAD_UNCHANGED)
                im2 = cv2.imread(pattern.format(pid, sid + 1), cv2.IMREAD_UNCHANGED)
                img = np.stack([im1, im2], axis=-1)
                img = img[off_y:off_y + crop_size[0], off_x:off_x + crop_size[1]]
                img = cv2.resize(img, (config.im_height, config.im_width), interpolation=cv2.INTER_LINEAR)
                eval_batch["im1"][bs] = img[:, :, 0]
                eval_batch["im2"][bs] = img[:, :, 1]
                bs = (bs + 1) % batch_size
                if bs == 0:
                    eval_batch["mirror"] = 0
                    yield copy.copy(eval_batch), labels
                    if config.eval_mirror:
                        if config.random_flip & 1 > 0:
                            tmp = copy.copy(eval_batch)
                            tmp["images"] = np.flip(tmp["images"], axis=2)
                            tmp["mirror"] = 1
                            yield tmp, None
                        if config.random_flip & 2 > 0:
                            tmp = copy.copy(eval_batch)
                            tmp["images"] = np.flip(tmp["images"], axis=1)
                            tmp["mirror"] = 2
                            yield tmp, None
                        if config.random_flip & 3 > 0:
                            tmp = copy.copy(eval_batch)
                            tmp["images"] = np.flip(np.flip(tmp["images"], axis=2), axis=1)
                            tmp["mirror"] = 3
                            yield tmp, None


def log_dataset(data_list, prefix=""):
    positives = 0
    negatives = 0
    for ci, case in enumerate(data_list):
        size = case["size"]
        for tumor in case["tumors"]:
            if tumor[3] - tumor[0] < 3:
                negatives += tumor[3] - tumor[0]
            else:
                negatives += 2
                positives += tumor[3] - tumor[0] - 2
                if tumor[0] == 0:
                    negatives -= 1
                if tumor[3] == size[0] - 1:
                    negatives -= 1
    logging.info("{}Total samples: {} = {}(Pos) + {}(Neg)".format(prefix, positives + negatives, positives, negatives))
