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
pattern = str(PROJ_ROOT / "data/NF/png/volume-{:d}/{:03d}_im.png")
lb_pattern = str(PROJ_ROOT / "data/NF/png/volume-{:d}/{:03d}_lb.png")
GRAY_MIN = 0
GRAY_MAX = 1000
TUMOR_PERCENT = 0.5
RND_SCALE = (1.0, 1.25)


def add_arguments(parser):
    group = parser.add_argument_group(title="Input Pipeline Arguments")
    group.add_argument("--test_fold", type=int, default=2)
    group.add_argument("--im_height", type=int, default=256)
    group.add_argument("--im_width", type=int, default=256)
    group.add_argument("--im_channel", type=int, default=3)
    group.add_argument("--filter_size", type=int, default=0, help="Filter tumors small than the given size")
    group.add_argument("--noise_scale", type=float, default=0.1)
    group.add_argument("--zoom_scale", type=float, nargs=2, default=RND_SCALE)
    group.add_argument("--random_flip", type=int, default=1,
                       help="Random flip while training. 0 for no flip, 1 for flip only left/right, "
                            "2 for only up/down, 3 for left/right and up/down")
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
        src_meta = Path(__file__).parent.parent.parent / "data/NF/png/meta.json"
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


def _get_test_data():
    prepare_dir = Path(__file__).parent / "prepare"
    prepare_dir.mkdir(parents=True, exist_ok=True)

    # Check existence
    obj_file = prepare_dir / "test_meta_update.json"
    if not obj_file.exists():
        raise FileNotFoundError("Cannot find test_meta_update.json")

    with obj_file.open() as f:
        dataset_dict = json.load(f)

    return dataset_dict


def _collect_datasets(test_fold, mode, filter_tumor_size=0, filter_only_liver_in_val=True):
    dataset_dict = _get_datasets(test_fold, filter_size=filter_tumor_size) if mode != "infer" else _get_test_data()
    if mode == "train":
        return dataset_dict["train"]
    elif mode == "infer":
        return dataset_dict["infer"]
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
                                filter_tumor_size=args.filter_size,
                                filter_only_liver_in_val=params.get("filter_only_liver_in_val", True))
    if len(dataset) == 0:
        raise ValueError("No valid dataset found!")

    with tf.variable_scope("InputPipeline"):
        if mode == ModeKeys.TRAIN:
            return get_dataset_for_train(dataset,
                                         tumor_percent=TUMOR_PERCENT,
                                         random_scale=args.zoom_scale, config=args)
        elif mode == "eval_online":
            return get_dataset_for_eval_online(dataset,
                                               tumor_percent=TUMOR_PERCENT,
                                               config=args)
        elif mode == ModeKeys.EVAL:
            if args.eval_in_patches:
                return get_dataset_for_eval_patches(dataset, config=args)
            else:
                return get_dataset_for_eval_image(dataset, config=args)
        elif mode == ModeKeys.PREDICT:
            return


#####################################
#
#   Input pipeline
#
#####################################


def data_processing_train(im_files, seg_file, read_size, bbox, PID_ci,  img_clip, config,
                          random_noise, random_flip_left_right, random_flip_up_down):
    off_x, off_y, height, width = bbox[0], bbox[1], bbox[2], bbox[3]

    def parse_im(name):
        return tf.cond(tf.greater(tf.strings.length(name), 0),
                       lambda: tf.image.decode_png(tf.io.read_file(name), channels=1, dtype=tf.uint16),
                       lambda: tf.zeros(read_size, dtype=tf.uint16))

    img = tf.map_fn(parse_im, im_files, dtype=tf.uint16)
    img = tf.image.crop_to_bounding_box(img, off_x, off_y, height, width)
    img = tf.image.resize_bilinear(img, (config.im_height, config.im_width), align_corners=True)
    img = tf.transpose(tf.squeeze(img, axis=-1), perm=(1, 2, 0))
    img = tf.cast(img, tf.float32)
    clip_min, clip_max = tf.split(img_clip, 2)
    img = (tf.clip_by_value(img, clip_min, clip_max) - clip_min) / (clip_max - clip_min)

    seg = tf.cond(tf.greater(tf.strings.length(seg_file), 0),
                  lambda: tf.image.decode_png(tf.io.read_file(seg_file), dtype=tf.uint8),
                  lambda: tf.zeros(read_size, dtype=tf.uint8))
    seg = tf.image.crop_to_bounding_box(seg, off_x, off_y, height, width)
    seg = tf.expand_dims(seg, axis=0)
    seg = tf.image.resize_nearest_neighbor(seg, (config.im_height, config.im_width), align_corners=True)
    seg = tf.cast(seg, tf.int32)
    seg = tf.squeeze(tf.squeeze(seg, axis=-1), axis=0)

    features = {"images": img, "names": PID_ci}
    labels = seg

    if random_noise:
        features["images"] = image_ops.random_noise(features["images"], config.noise_scale)
        # Remove noise in empty slice
        features["images"] *= tf.cast(tf.greater(tf.strings.length(im_files), 0), tf.float32)
        logging.info("Train: Add random noise, scale = {}".format(config.noise_scale))
    if random_flip_left_right:
        features["images"], labels = image_ops.random_flip_left_right(features["images"], labels)
        logging.info("Train: Add random flip left <-> right")
    if random_flip_up_down:
        features["images"], labels = image_ops.random_flip_up_down(features["images"], labels)
        logging.info("Train: Add random flip up <-> down")

    return features, labels


def gen_train_batch(data_list,
                    batch_size,
                    tumor_percent=0.,
                    random_scale=(1., 1.),
                    random_window_level=False,
                    config=None):
    d = data_list
    list_of_keys = np.arange(len(d))
    tumor_list_of_keys = []
    for i in list_of_keys:
        if len(d[i]["slices"]) > 0:
            tumor_list_of_keys.append(i)
    target_size = np.asarray((config.im_height, config.im_width), dtype=np.float32)
    force_tumor = math.ceil(batch_size * tumor_percent)

    while True:
        ci1 = np.random.choice(tumor_list_of_keys, force_tumor, True)
        ci2 = np.random.choice(list_of_keys, batch_size - force_tumor, True)   # case indices
        ci = np.concatenate((ci1, ci2), axis=0)

        tumor_counter = 0
        for j, i in enumerate(ci):
            case = d[i]
            crop_size = list((target_size * np.random.uniform(*random_scale, size=2)).astype(np.int32))
            size = case["size"]
            pid = case["PID"]
            patch_size = [size[1], size[2], 1]

            # Get selected slice
            if tumor_counter < force_tumor:
                tumor_slices = case["slices"]
                ind = np.random.choice(np.arange(len(tumor_slices)))
                selected_slice = case["tumor_slices_index"][ind]
                tumor_counter += 1
                obj_bb = tumor_slices[ind][random.randint(0, len(tumor_slices[ind]) - 1)]
            else:
                selected_slice = random.randint(0, size[0] - 1)
                obj_bb = [size[1], size[2], 0, 0]   # Obj not exist

            # Compute crop region
            rng_yl = max(obj_bb[2] + 5 - crop_size[0], 0)
            rng_yr = min(obj_bb[0] - 5, size[1] - crop_size[0])
            rng_xl = max(obj_bb[3] + 5 - crop_size[1], 0)
            rng_xr = min(obj_bb[1] - 5, size[2] - crop_size[1])
            off_y = random.randint(rng_yl, rng_yr)
            off_x = random.randint(rng_xl, rng_xr)

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
                        # slice index is "" means padding with zeros
                        selected_slices.append("")

            # Random clip image value
            if random_window_level:
                img_clip = (0, random.randint(800, 1000))
            else:
                img_clip = (0, 900)

            yield selected_slices, lb_pattern.format(pid, selected_slice), patch_size, \
                [off_y, off_x] + crop_size, pid, img_clip


def get_dataset_for_train(data_list, tumor_percent=0., random_scale=(1., 1.), config=None):
    batch_size = distribution_utils.per_device_batch_size(config.batch_size, config.num_gpus)
    if random_scale[1] > random_scale[0]:
        logging.info("Train: Add random zoom, scale = ({}, {})".format(*random_scale))

    def train_gen():
        return gen_train_batch(data_list, batch_size, tumor_percent,
                               random_scale, random_window_level=True, config=config)

    dataset = (tf.data.Dataset.from_generator(train_gen,
                                              (tf.string, tf.string, tf.int32, tf.int32, tf.int32, tf.float32),
                                              output_shapes=(tf.TensorShape([config.im_channel]),
                                                             tf.TensorShape([]),
                                                             tf.TensorShape([4]),
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


def get_dataset_for_eval_online(data_list, tumor_percent=0., config=None):
    batch_size = distribution_utils.per_device_batch_size(config.batch_size, config.num_gpus)

    def val_gen():
        infinity_generator = gen_train_batch(data_list, batch_size, tumor_percent,
                                             random_scale=(1., 1.), config=config)
        for _ in tqdm.tqdm(range(config.eval_num_batches_per_epoch * config.batch_size)):
            yield next(infinity_generator)

    dataset = (tf.data.Dataset.from_generator(val_gen, (tf.string, tf.string, tf.int32, tf.int32, tf.int32, tf.float32),
                                              output_shapes=(tf.TensorShape([config.im_channel]),
                                                             tf.TensorShape([]),
                                                             tf.TensorShape([4]),
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


def data_processing_eval(img, x1, y1, x2, y2, dsize, im_scale):
    img = img[y1:y2, x1:x2]
    if (y2 - y1, x2 - x1) != dsize:
        img = cv2.resize(img, dsize, interpolation=cv2.INTER_LINEAR)
    return img / im_scale


def parse_case(case):
    pid = case["PID"]
    d, h, w = case["size"]

    return pid, d, h, w


def parse_case_eval(case, im_channel, parse_label=True):
    """ Return cropped normalized volume (y, x, z) with type float32 and
               cropped segmentation (z, y, x) with type uint8 """
    d, h, w = case["size"]

    obj_num = int(case["vol_case"][:-4].split("-")[-1])
    _, volume = nii_kits.read_lits(obj_num, "vol", PROJ_ROOT / case["vol_case"])
    left_half_channel = (im_channel - 1) // 2
    right_half_channel = im_channel - 1 - left_half_channel
    volume = np.concatenate((np.zeros((left_half_channel, h, w), dtype=volume.dtype),
                             volume,
                             np.zeros((right_half_channel, h, w), dtype=volume.dtype)), axis=0)
    cd, ch, cw = volume.shape
    volume = (np.clip(volume, GRAY_MIN, GRAY_MAX) - GRAY_MIN) / (GRAY_MAX - GRAY_MIN)
    volume = volume.transpose((1, 2, 0)).astype(np.float32)  # (y, x, z) for convenient

    segmentation = None
    lab_case = None
    if parse_label:
        _, segmentation = nii_kits.read_lits(obj_num, "lab", PROJ_ROOT / case["lab_case"])
        segmentation = segmentation.astype(np.uint8)
        lab_case = case["lab_case"]

    oshape = [d, h, w]
    cshape = [cd, ch, cw]
    return case["PID"], case["vol_case"], lab_case, oshape, cshape, \
        left_half_channel, right_half_channel, volume, segmentation


def get_dataset_for_eval_image(data_list, config=None):
    batch_size = config.batch_size
    c = config.im_channel
    pshape = config.im_height, config.im_width
    resize = True
    if config.im_height <= 0 or config.im_width <= 0:
        logging.info("Disable image resize for evaluating")
        resize = False

    for ci, case in enumerate(data_list[config.eval_skip_num:]):
        pid, vol_path, _, oshape, cshape, lhc, rhc, volume, segmentation = \
            parse_case_eval(case, c, parse_label=config.mode != ModeKeys.PREDICT)
        if not resize:
            pshape = cshape[1:]

        eval_batch = {"images": np.empty((batch_size, *pshape, c), dtype=np.float32),
                      "names": pid,
                      "mirror": 0}

        pads = (batch_size - (cshape[0] % batch_size)) % batch_size
        if pads > 0:
            volume = np.concatenate((volume, np.zeros((*cshape[1:], pads), volume.dtype)), axis=-1)
        if resize:
            volume = cv2.resize(volume, pshape, interpolation=cv2.INTER_LINEAR)

        num_of_batches = (volume.shape[-1] - lhc - rhc) // batch_size
        assert volume.shape[-1] - lhc - rhc == batch_size * num_of_batches, \
            "Wrong padding: volume length: {}, lhc: {}, rhc: {}, batch_size: {}, num_of_batches: {}".format(
                volume.shape[-1], lhc, rhc, batch_size, num_of_batches)
        for idx in range(lhc, volume.shape[-1] - rhc, batch_size):
            for j in range(batch_size):
                eval_batch["images"][j] = volume[:, :, idx + j - lhc:idx + j + rhc + 1]
            yield copy.copy(eval_batch), None

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
        yield None, (segmentation, vol_path, pads, resize)


def get_dataset_for_eval_patches(data_list, step=2, config=None):
    """
    Parameters
    ----------
    data_list
    step: int
        how large is the step size for sliding window? 2 := patch_size // 2 for each axis
    config
    """
    batch_size = config.batch_size
    c = config.im_channel
    psize = config.im_height, config.im_width

    for ci, case in enumerate(data_list[config.eval_skip_num:]):
        pid, vol_path, _, oshape, cshape, lhc, rhc, volume, segmentation = \
            parse_case_eval(case, c, parse_label=config.mode != ModeKeys.PREDICT)

        eval_batch = {"images": np.empty((batch_size, *psize, c),
                                         dtype=np.float32),
                      "name": pid,
                      "pad": 0,
                      "position": [None for _ in range(batch_size)]}

        center_start = np.array([psize[0] // 2, psize[1] // 2]).astype(np.int32)
        center_end = np.array([cshape[1] - psize[0] // 2, cshape[2] - psize[1] // 2]).astype(np.int32)
        num_steps = np.ceil([(center_end[i] - center_start[i]) / (psize[i] / step) for i in range(2)])
        step_size = np.array([(center_end[i] - center_start[i]) / (num_steps[i] + 1e-8) for i in range(2)])
        step_size[step_size == 0] = 9999999
        ysteps = np.round(np.arange(center_start[0], center_end[0] + 1e-8, step_size[0])).astype(np.int32)
        xsteps = np.round(np.arange(center_start[1], center_end[1] + 1e-8, step_size[1])).astype(np.int32)

        all_patches = list(itertools.product(xsteps, ysteps, range(lhc, cshape[0] - rhc)))
        num_of_batches = (len(all_patches) + (batch_size - 1)) // batch_size

        for batch in range(num_of_batches - 1):
            start_id = batch * batch_size
            for i, (x, y, z) in enumerate(all_patches[start_id:start_id + batch_size]):
                lb_x = x - psize[0] // 2
                ub_x = x + psize[0] // 2
                lb_y = y - psize[1] // 2
                ub_y = y + psize[1] // 2
                lb_c = z - lhc
                ub_c = z + rhc + 1
                eval_batch["images"][i] = volume[lb_y:ub_y, lb_x:ub_x, lb_c:ub_c]
                eval_batch["position"][i] = (z - lhc, lb_y, ub_y, lb_x, ub_x)
            yield copy.copy(eval_batch), None

        # Final batch
        start_id = (num_of_batches - 1) * batch_size
        for i, (x, y, z) in enumerate(all_patches[start_id:]):
            lb_x = x - psize[0] // 2
            ub_x = x + psize[0] // 2
            lb_y = y - psize[1] // 2
            ub_y = y + psize[1] // 2
            lb_c = z - lhc
            ub_c = z + rhc + 1
            eval_batch["images"][i] = volume[lb_y:ub_y, lb_x:ub_x, lb_c:ub_c]
            eval_batch["position"][i] = (z - lhc, lb_y, ub_y, lb_x, ub_x)
        rem = len(all_patches) % batch_size
        if rem > 0:
            eval_batch["images"][rem:] = 0
        eval_batch["pad"] = (batch_size - rem) % batch_size

        yield copy.copy(eval_batch), segmentation


if __name__ == "__main__":
    # gen_dataset_jsons()
    # pattern = str(Path(__file__).parent.parent.parent / "data/LiTS/png/volume-{:d}/{:03d}_im.png")
    # lb_pattern = str(Path(__file__).parent.parent.parent / "data/LiTS/png/volume-{:d}/{:03d}_lb.png")
    pattern = str(Path("D:/DataSet") / "LiTS/png/volume-{:d}/{:03d}_im.png")
    lb_pattern = str(Path("D:/DataSet") / "LiTS/png/volume-{:d}/{:03d}_lb.png")
    import matplotlib
    matplotlib.use("Qt5Agg")
    import matplotlib.pyplot as plt

    class Foo(object):
        batch_size = 8
        im_height = 256
        im_width = 256
        im_channel = 3
        num_gpus = 1
        noise_scale = 0.05

    cnt = 0

    def save(f, l, cnt, lst):
        if lst is None or lst != f["names"][0]:
            cnt = 0
        for i in range(8):
            plt.subplot(231)
            plt.imshow(f["images"][i, :, :, 0], cmap="gray")
            plt.subplot(232)
            plt.imshow(f["images"][i, :, :, 1], cmap="gray")
            plt.subplot(233)
            plt.imshow(f["images"][i, :, :, 2], cmap="gray")
            plt.subplot(235)
            plt.imshow(l[i, :, :], cmap="gray")
            plt.savefig(r"D:\downloads\temp\{}-{}.png".format(f["names"][0], cnt))
            plt.close()
            cnt += 1
        return cnt, f["names"][0]

    def show(f=None, l=None, idx=0):
        if f is not None:
            plt.subplot(231)
            plt.imshow(f["images"][idx, :, :, 0], cmap="gray")
            plt.subplot(232)
            plt.imshow(f["images"][idx, :, :, 1], cmap="gray")
            plt.subplot(233)
            plt.imshow(f["images"][idx, :, :, 2], cmap="gray")
        if l is not None:
            plt.subplot(235)
            plt.imshow(l[idx, :, :], cmap="gray")
        plt.show()

    data_list = _get_datasets(choices=[0, 31])["choices"]
    gen = get_dataset_for_eval_patches(data_list, config=Foo())
    # lst = None
    # while True:
    #     try:
    #         f, l = next(gen)
    #         cnt, lst = save(f, l, cnt, lst)
    #     except StopIteration:
    #         break
    f, l = next(gen)
    show(f, l, 0)

    # d = get_dataset_for_train(data_list, liver_percent=0.66, tumor_percent=0.4,
    #                           random_scale=(1., 1.4), args=Foo())
    # ff, ll = d.make_one_shot_iterator().get_next()
    # sess = tf.Session()
    # f, l = sess.run([ff, ll])
    # show(f, l, 0)
