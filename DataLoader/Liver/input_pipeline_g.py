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
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import tensorflow_estimator as tfes
from tensorflow.python.platform import tf_logging as logging
from tensorflow.contrib import data as contrib_data

from utils import image_ops
from utils import distribution_utils
from DataLoader import misc
from DataLoader import feature_ops
from DataLoader.Liver import nii_kits

ModeKeys = tfes.estimator.ModeKeys
Dataset = tf.data.Dataset
pattern = str(Path(__file__).parent.parent.parent / "data/LiTS/png/volume-{:d}/{:03d}_im.png")
lb_pattern = str(Path(__file__).parent.parent.parent / "data/LiTS/png/volume-{:d}/{:03d}_lb.png")
IM_SCALE = 128 * 450
LB_SCALE = 64
GRAY_MIN = -200
GRAY_MAX = 250
LIVER_PERCENT = 0.66
TUMOR_PERCENT = 0.5
RND_SCALE = (1., 1.5)


def add_arguments(parser):
    group = parser.add_argument_group(title="Input Pipeline Arguments")
    group.add_argument("--test_fold", type=int, default=2)
    group.add_argument("--im_height", type=int, default=256)
    group.add_argument("--im_width", type=int, default=256)
    group.add_argument("--im_channel", type=int, default=3)
    group.add_argument("--noise_scale", type=float, default=0.1)
    group.add_argument("--eval_in_patches", action="store_true")
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
        src_meta = Path(__file__).parent.parent.parent / "data/LiTS/png/meta.json"
        if not src_meta.exists():
            raise FileNotFoundError(str(src_meta))
        shutil.copyfile(str(src_meta), str(meta_file))
    with meta_file.open() as f:
        meta = json.load(f)

    def parse(case):
        case.pop("tumors")
        case.pop("tumor_areas")
        case.pop("tumor_centers")
        case.pop("tumor_stddevs")
        ft = case.pop("tumor_slices_from_to")
        length = len(ft) - 1
        assert len(ft) == len(case["tumor_slices_index"]) + 1
        centers = case.pop("tumor_slices_centers")
        stddevs = case.pop("tumor_slices_stddevs")
        areas = case.pop("tumor_slices_areas")
        coords = case.pop("tumor_slices")
        case["centers"] = []
        case["stddevs"] = []
        case["slices"] = []
        slices = copy.deepcopy(case["tumor_slices_index"])
        for ii in range(length):
            c = centers[ft[ii]:ft[ii + 1]]
            s = stddevs[ft[ii]:ft[ii + 1]]
            a = areas[ft[ii]:ft[ii + 1]]
            o = coords[ft[ii]:ft[ii + 1]]
            select = np.where(np.asarray(a) > filter_size)[0]
            if select.shape[0] == 0:
                case["tumor_slices_index"].remove(slices[ii])
            else:
                case["centers"].append([c[j] for j in select])
                case["stddevs"].append([s[j] for j in select])
                case["slices"].append([o[j] for j in select])
        return case

    if not choices:
        # Load k_folds
        fold_path = prepare_dir / "k_folds.txt"
        all_cases = list(range(131))
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


def _collect_datasets(test_fold, mode, filter_only_liver_in_val=True):
    dataset_dict = _get_datasets(test_fold, filter_size=0)
    if mode == "train":
        return dataset_dict["train"]
    else:
        if not filter_only_liver_in_val:
            return dataset_dict["val"]
        else:
            new_dict = []
            for case in dataset_dict["val"]:
                if len(case["slices"]) > 0:
                    new_dict.append(case)
            return new_dict


def input_fn(mode, params):
    if "args" not in params:
        raise KeyError("params of input_fn need an \"args\" key")

    args = params["args"]
    dataset = _collect_datasets(args.test_fold, mode,
                                filter_only_liver_in_val=params.get("filter_only_liver_in_val", True))
    if len(dataset) == 0:
        raise ValueError("No valid dataset found!")

    with tf.variable_scope("InputPipeline"):
        if mode == ModeKeys.TRAIN:
            return get_dataset_for_train(dataset,
                                         liver_percent=LIVER_PERCENT,
                                         tumor_percent=TUMOR_PERCENT,
                                         random_scale=RND_SCALE, config=args)
        elif mode == "eval_online":
            return get_dataset_for_eval_online(dataset,
                                               liver_percent=LIVER_PERCENT,
                                               tumor_percent=TUMOR_PERCENT,
                                               config=args)
        elif mode == ModeKeys.EVAL:
            if args.eval_in_patches:
                return get_dataset_for_eval_patches(dataset, config=args)
            else:
                return get_dataset_for_eval_image(dataset, config=args)

#####################################
#
#   G-Net input pipeline
#
#####################################


def data_processing_train(im_files, seg_file, bbox, PID_ci, guide, img_scale, lab_scale, config,
                          random_noise, random_flip_left_right, random_flip_up_down, **kwargs):
    off_x, off_y, height, width = bbox[0], bbox[1], bbox[2], bbox[3]

    def parse_im(name):
        return tf.cond(tf.greater(tf.strings.length(name), 0),
                       lambda: tf.image.decode_png(tf.io.read_file(name), channels=1, dtype=tf.uint16),
                       lambda: tf.zeros((512, 512, 1), dtype=tf.uint16))

    img = tf.map_fn(parse_im, im_files, dtype=tf.uint16)
    img = tf.image.crop_to_bounding_box(img, off_x, off_y, height, width)
    img = tf.image.resize_bilinear(img, (config.im_height, config.im_width), align_corners=True)
    img = tf.transpose(tf.squeeze(img, axis=-1), perm=(1, 2, 0))
    img = tf.cast(img, tf.float32) / img_scale

    seg = tf.image.decode_png(tf.io.read_file(seg_file), dtype=tf.uint8)
    seg = tf.image.crop_to_bounding_box(seg, off_x, off_y, height, width)
    seg = tf.expand_dims(seg, axis=0)
    seg = tf.image.resize_nearest_neighbor(seg, (config.im_height, config.im_width), align_corners=True)
    seg = tf.cast(seg / lab_scale, tf.int32)
    seg = tf.squeeze(tf.squeeze(seg, axis=-1), axis=0)

    features = {"images": img, "names": PID_ci}
    labels = seg

    if random_noise:
        features["images"] = image_ops.random_noise(features["images"], config.noise_scale)
        logging.info("Train: Add random noise, scale = {}".format(config.noise_scale))
    if random_flip_left_right:
        features["images"], labels = image_ops.random_flip_left_right(features["images"], labels)
        logging.info("Train: Add random flip left <-> right")
    if random_flip_up_down:
        features["images"], labels = image_ops.random_flip_up_down(features["images"], labels)
        logging.info("Train: Add random flip up <-> down")

    if "context" in guide:
        features["context"] = guide["context"]
        logging.info("Train: Use context guide")

    def true_fn():
        gd = image_ops.create_spatial_guide_2d(guide["crop_size"], guide["centers"], guide["stddevs"])
        gd_resize = tf.image.resize_bilinear(gd, (config.im_height, config.im_width), align_corners=True)
        return tf.cast(gd_resize / 2 + 0.5, tf.float32)

    def false_fn():
        return tf.ones((config.im_height, config.im_width), tf.float32) * 0.5

    if "centers" in guide:
        features["sp_guide"] = tf.cond(tf.shape(guide["centers"])[0] > 0, true_fn, false_fn)
        logging.info("Train: Use spatial guide")

    return features, labels


def gen_train_batch(data_list,
                    batch_size,
                    liver_percent=0.,
                    tumor_percent=0.,
                    random_scale=(1., 1.),
                    context_guide=None,
                    context_list=(("hist", 200),),
                    spatial_guide=None,
                    spatial_random=0.,
                    config=None,
                    **kwargs):
    # Load context guide if needed
    context = {case["PID"]: None for case in data_list} if context_guide else None

    d = data_list
    list_of_keys = np.arange(len(d))
    tumor_list_of_keys = []
    for i in list_of_keys:
        if len(d[i]["slices"]) > 0:
            tumor_list_of_keys.append(i)
    target_size = np.asarray((config.im_height, config.im_width), dtype=np.float32)
    force_liver = math.ceil(batch_size * liver_percent)
    force_tumor = math.ceil(batch_size * tumor_percent)
    # empty_guide = np.ones((args.im_height, args.im_width, 1), dtype=np.float32) * 0.5 \
    #     if spatial_guide else None
    empty_mmts = np.zeros((0, 2), dtype=np.float32)
    if random_scale[1] > random_scale[0]:
        logging.info("Train: Add random zoom, scale = ({}, {})".format(*random_scale))

    while True:
        ci1 = np.random.choice(tumor_list_of_keys, force_tumor, True)
        ci2 = np.random.choice(list_of_keys, batch_size - force_tumor, True)  # case indices
        ci = np.concatenate((ci1, ci2), axis=0)

        liver_counter = 0
        tumor_counter = 0
        for j, i in enumerate(ci):
            case = d[i]
            crop_size = (target_size * random.uniform(*random_scale)).astype(np.int32).tolist()
            size = case["size"]
            pid = case["PID"]

            # Get selected slice
            if tumor_counter < force_tumor:
                tumor_slices = case["slices"]
                ind = np.random.choice(np.arange(len(tumor_slices)))
                selected_slice = case["tumor_slices_index"][ind]
                tumor_counter += 1
                liver_counter += 1
                obj_bb = tumor_slices[ind][random.randint(0, len(tumor_slices[ind]) - 1)]
            elif liver_counter < force_liver:
                selected_slice = random.randint(case["bbox"][0], case["bbox"][3] - 1)
                liver_counter += 1
                obj_bb = case["bbox"][1:3] + case["bbox"][4:6]
                # Record tumor slice indices for spatial guide
                if selected_slice in case["tumor_slices_index"]:
                    ind = case["tumor_slices_index"].index(selected_slice)
                else:
                    ind = -1
            else:
                selected_slice = random.randint(0, size[0] - 1)
                obj_bb = [size[1], size[2], 0, 0]   # Obj not exist
                if selected_slice in case["tumor_slices_index"]:
                    ind = case["tumor_slices_index"].index(selected_slice)
                else:
                    ind = -1

            # Compute crop region
            rng_yl = max(obj_bb[2] - crop_size[0], 0)
            rng_yr = min(obj_bb[0], size[1] - crop_size[0])
            if rng_yl + 20 < rng_yr:
                off_y = random.randint(rng_yl, rng_yr)
            else:
                # obj_bbox size exceeds crop_size or less than 20 pixels for random choices,
                # we will crop part of object
                off_y = random.randint(max(obj_bb[0] - 20, 0),
                                       min(int(obj_bb[0] * .75 + obj_bb[2] * .25), size[1] - crop_size[0]))
            rng_xl = max(obj_bb[3] - crop_size[1], 0)
            rng_xr = min(obj_bb[1], size[2] - crop_size[1])
            if rng_xl + 20 < rng_xr:
                off_x = random.randint(rng_xl, rng_xr)
            else:
                off_x = random.randint(max(obj_bb[1] - 20, 0),
                                       min((obj_bb[1] + obj_bb[3]) // 2, size[2] - crop_size[1]))

            # Get multi-channel input
            selected_slices = [pattern.format(pid, selected_slice)]
            if config.im_channel > 1:
                left_half_channel = (config.im_channel - 1) // 2
                for k in range(1, left_half_channel + 1):
                    previous_slice = selected_slice - k
                    if previous_slice >= 0:
                        selected_slices.insert(0, pattern.format(pid, previous_slice))
                    else:
                        # slice index is "" means padding with zeros
                        selected_slices.insert(0, "")
                right_half_channel = config.im_channel - 1 - left_half_channel
                for k in range(1, right_half_channel + 1):
                    following_slice = selected_slice + k
                    if following_slice < size[0]:
                        selected_slices.append(pattern.format(pid, following_slice))
                    else:
                        selected_slices.append("")

            # Last element store context guide and spatial guide information
            yield_list = (selected_slices, lb_pattern.format(pid, selected_slice),
                          [off_y, off_x] + crop_size, pid, {})

            # context guide
            if context_guide:
                # Load texture features when needed
                if context[pid] is None:
                    gd_pattern = Path(__file__).parent.parent.parent / "data/LiTS/feat"
                    gd_pattern = str(gd_pattern.parent / "%s" / "train" / "%03d.npy")
                    features = []
                    for cls, f_len in context_list:
                        feat = np.load(gd_pattern % (cls, pid), allow_pickle=True)
                        assert isinstance(feat, np.ndarray), "`feat` must be a numpy.ndarray"
                        assert feat.shape[1] == f_len, "feature length mismatch %d vs %d" \
                                                       % (feat.shape[1], f_len)
                        features.append(eval("feature_ops.%d_aug" % cls)(feat, **kwargs))
                    context[pid] = np.concatenate(features, axis=1).astype(np.float32)
                yield_list[-1]["context"] = context[pid][selected_slice]

            # spatial guide
            if spatial_guide:
                inbox_tumors = []
                use_spatial_guide = random.random() < spatial_random
                if use_spatial_guide and ind >= 0:
                    centers = np.asarray(case["centers"][ind], dtype=np.float32)
                    stddevs = np.asarray(case["stddevs"][ind], dtype=np.float32)
                    num_of_tumors = centers.shape[0]
                    tumor_choose = list(range(num_of_tumors))
                    for tc in tumor_choose:
                        if off_y <= centers[tc][0] < off_y + crop_size[0] and \
                                off_x <= centers[tc][1] < off_x + crop_size[1]:
                            inbox_tumors.append(tc)
                if inbox_tumors:
                    inbox_choose = random.sample(inbox_tumors, k=random.randint(1, len(inbox_tumors)))
                    yield_list[-1].update({"centers": centers[inbox_choose] - crop_size,
                                           "stddevs": stddevs[inbox_choose],
                                           "crop_size": crop_size})
                else:
                    yield_list[-1].update({"centers": empty_mmts,
                                           "stddevs": empty_mmts,
                                           "crop_size": crop_size})

            yield yield_list


def get_dataset_for_train(data_list,
                          liver_percent=0.,
                          tumor_percent=0.,
                          random_scale=(1., 1.),
                          context_guide=None,
                          context_list=(),
                          spatial_guide=None,
                          spatial_random=0.,
                          config=None,
                          **kwargs):
    batch_size = distribution_utils.per_device_batch_size(config.batch_size, config.num_gpus)

    def train_gen():
        return gen_train_batch(data_list, batch_size, liver_percent, tumor_percent,
                               random_scale=random_scale,
                               context_guide=context_guide,
                               context_list=context_list,
                               spatial_guide=spatial_guide,
                               spatial_random=spatial_random,
                               config=config,
                               **kwargs)

    guide_types_dict = {}
    guide_shapes_dict = {}
    if context_guide:
        guide_types_dict["context"] = tf.float32
        feature_length = sum([x for _, x in context_list])
        guide_shapes_dict["context"] = tf.TensorShape([feature_length])
    if spatial_guide:
        guide_types_dict.update({"centers": tf.float32,
                                 "stddevs": tf.float32,
                                 "crop_size": tf.int32})
        guide_shapes_dict.update({"centers": tf.TensorShape([None, 2]),
                                  "stddevs": tf.TensorShape([None, 2]),
                                  "crop_size": tf.TensorShape([2])})

    output_types = (tf.string, tf.string, tf.int32, tf.int32, guide_types_dict)
    output_shapes = (tf.TensorShape([config.im_channel]),
                     tf.TensorShape([]),
                     tf.TensorShape([4]),
                     tf.TensorShape([]),
                     guide_shapes_dict)

    dataset = (tf.data.Dataset.from_generator(train_gen, output_types, output_shapes)
               .apply(tf.data.experimental.map_and_batch(
                        lambda *args_: data_processing_train(*args_,
                                                             img_scale=IM_SCALE,
                                                             lab_scale=LB_SCALE,
                                                             config=config,
                                                             random_noise=True,
                                                             random_flip_left_right=True,
                                                             random_flip_up_down=True),
                        batch_size, num_parallel_batches=1))
               .prefetch(buffer_size=contrib_data.AUTOTUNE))

    return dataset


def get_dataset_for_eval_online(data_list,
                                liver_percent=0.,
                                tumor_percent=0.,
                                context_guide=None,
                                context_list=(),
                                spatial_guide=None,
                                config=None,
                                **kwargs):
    batch_size = distribution_utils.per_device_batch_size(config.batch_size, config.num_gpus)

    def train_gen():
        return gen_train_batch(data_list, batch_size, liver_percent, tumor_percent,
                               random_scale=(1., 1.),
                               context_guide=context_guide,
                               context_list=context_list,
                               spatial_guide=spatial_guide,
                               spatial_random=0.,
                               config=config,
                               **kwargs)

    guide_types_dict = {}
    guide_shapes_dict = {}
    if context_guide:
        guide_types_dict["context"] = tf.float32
        feature_length = sum([x for _, x in context_list])
        guide_shapes_dict["context"] = tf.TensorShape([feature_length])
    if spatial_guide:
        guide_types_dict.update({"centers": tf.float32,
                                 "stddevs": tf.float32,
                                 "crop_size": tf.int32})
        guide_shapes_dict.update({"centers": tf.TensorShape([None, 2]),
                                  "stddevs": tf.TensorShape([None, 2]),
                                  "crop_size": tf.TensorShape([2])})

    output_types = (tf.string, tf.string, tf.int32, tf.int32, guide_types_dict)
    output_shapes = (tf.TensorShape([config.im_channel]),
                     tf.TensorShape([]),
                     tf.TensorShape([4]),
                     tf.TensorShape([]),
                     guide_shapes_dict)

    dataset = (tf.data.Dataset.from_generator(train_gen, output_types, output_shapes)
               .apply(tf.data.experimental.map_and_batch(
                        lambda *args_: data_processing_train(*args_,
                                                             img_scale=IM_SCALE,
                                                             lab_scale=LB_SCALE,
                                                             config=config,
                                                             random_noise=False,
                                                             random_flip_left_right=False,
                                                             random_flip_up_down=False),
                        batch_size, num_parallel_batches=1))
               .prefetch(buffer_size=contrib_data.AUTOTUNE))

    return dataset
