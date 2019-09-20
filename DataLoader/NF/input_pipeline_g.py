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
import numpy as np
from pathlib import Path
from scipy import ndimage as ndi
import tensorflow as tf
import tensorflow_estimator as tfes
from tensorflow.python.platform import tf_logging as logging
from tensorflow.contrib import data as contrib_data
from tensorflow.python.util import deprecation

from utils import image_ops
from utils import array_kits
from utils import distribution_utils
from DataLoader import misc
# noinspection PyUnresolvedReferences
from DataLoader import feature_ops
from DataLoader.Liver import nii_kits

deprecation._PRINT_DEPRECATION_WARNINGS = False
ModeKeys = tfes.estimator.ModeKeys
Dataset = tf.data.Dataset
PROJ_ROOT = Path(__file__).parent.parent.parent
pattern = str(PROJ_ROOT / "data/NF/png/volume-{:03d}/{:03d}_im.png")
lb_pattern = str(PROJ_ROOT / "data/NF/png/volume-{:03d}/{:03d}_lb.png")
GRAY_MIN = 0
GRAY_MAX = 1000
TUMOR_PERCENT = 0.5
RND_SCALE = (1.0, 1.25)

RND_SEED = None     # For debug
# random.seed(1234)
# np.random.seed(1234)

# Pre-computed glcm noise scale
glcm_noise_scale = np.array([0.0004, 0.0007, 0.0003, 0.0007, 0.0014, 0.0007, 0.0009, 0.0007, 0.0023, 0.0021,
                             0.0015, 0.002 , 0.0014, 0.0018, 0.0011, 0.0017, 0.0026, 0.0018, 0.0019, 0.0017,
                             0.0033, 0.0031, 0.0026, 0.003 , 0.0033, 0.0027, 0.0041, 0.0028, 0.002 , 0.0027,
                             0.0025, 0.0028, 0.0015, 0.0017, 0.002 , 0.0017, 0.0015, 0.0015, 0.0015, 0.0015,
                             0.0015, 0.0015, 0.0015, 0.0015, 0.0015, 0.0016, 0.0015, 0.0016, 0.0072, 0.0072,
                             0.0072, 0.0072, 0.0073, 0.0072, 0.0073, 0.0072, 0.0071, 0.0071, 0.0072, 0.0071,
                             0.0083, 0.0073, 0.0089, 0.0074, 0.0053, 0.0073, 0.007 , 0.0074, 0.0032, 0.0034,
                             0.0053, 0.0036, 0.0026, 0.0021, 0.0027, 0.0021, 0.0019, 0.0021, 0.0021, 0.0021,
                             0.0015, 0.0012, 0.0016, 0.0012, 0.0417, 0.0394, 0.043 , 0.0396, 0.0358, 0.0394,
                             0.039 , 0.0396, 0.0305, 0.0314, 0.035 , 0.032 ], np.float32)


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

    group = parser.add_argument_group(title="G-Net Arguments")
    group.add_argument("--side_dropout", type=float, default=0.5, help="Dropout used in G-Net sub-networks")
    group.add_argument("--use_context", action="store_true")
    group.add_argument("--context_list", type=str, nargs="+",
                       help="Paired context information: (feature name, feature length). "
                            "For example: hist, 200")
    group.add_argument("--hist_noise", action="store_true",
                       help="Augment histogram dataset with random noise")
    group.add_argument("--hist_noise_scale", type=float, default=0.002,
                       help="Random noise scale for histogram (default: %(default)f)")
    group.add_argument("--hist_scale", type=float, default=20,
                       help="A coefficient multiplied to histogram values")
    group.add_argument("--glcm", action="store_true",
                       help="Use glcm texture features")
    group.add_argument("--glcm_noise", action="store_true")

    group.add_argument("--use_spatial", action="store_true")
    group.add_argument("--spatial_random", type=float, default=1.,
                       help="Probability of adding spatial guide to current slice with tumors "
                            "when use_spatial is on")
    group.add_argument("--spatial_inner_random", action="store_true",
                       help="Random choice tumors to give spatial guide inside a slice with tumors")
    group.add_argument("--center_random_ratio", type=float, default=0.2,
                       help="Random perturbation scale of the spatial guide centers")
    group.add_argument("--stddev_random_ratio", type=float, default=0.4,
                       help="Random perturbation scale of the spatial guide stddevs")
    group.add_argument("--eval_no_sp", action="store_true", help="No spatial guide in evaluation")
    group.add_argument("--min_std", type=float, default=2.,
                       help="Minimum stddev for spatial guide")
    group.add_argument("--save_sp_guide", action="store_true", help="Save spatial guide")
    group.add_argument("--use_se", action="store_true", help="Use SE-Block in G-Nets context guide module")
    group.add_argument("--eval_discount", type=float, default=0.85)
    group.add_argument("--eval_no_p", action="store_true", help="Evaluate with no propagation")
    group.add_argument("--real_sp", type=str, help="Path to real spatial guide.")


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
        if len(case["tumors"]) == 0:
            return
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
            case = parse(meta[idx])
            if case:
                dataset_dict["train"].append(case)
        for idx in sorted([int(x) for x in testset]):
            case = parse(meta[idx])
            if case:
                dataset_dict["val"].append(case)

        with obj_file.open("w") as f:
            json.dump(dataset_dict, f)
    else:
        dataset_dict = {"choices": []}
        for idx in choices:
            case = parse(meta[idx])
            if case:
                dataset_dict["choices"].append(case)

    return dataset_dict


def _collect_datasets(test_fold, mode, filter_tumor_size=0):
    dataset_dict = _get_datasets(test_fold, filter_size=filter_tumor_size)
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
    dataset = _collect_datasets(args.test_fold, mode, filter_tumor_size=args.filter_size)
    if len(dataset) == 0:
        raise ValueError("No valid dataset found!")
    logging.info("{}: {} NF MRIs ({} slices, {} slices contain tumors)"
                 .format(mode[:1].upper() + mode[1:], len(dataset),
                         sum([x["size"][0] for x in dataset]),
                         sum([len(x["tumor_slices_index"]) for x in dataset])))
    # Parse context_list
    context_list = []
    if args.use_context and args.context_list is not None:
        if len(args.context_list) % 2 != 0:
            raise ValueError("context_list is not paired!")
        for i in range(len(args.context_list) // 2):
            context_list.append((args.context_list[2 * i], int(args.context_list[2 * i + 1])))
    features = [x for x, y in context_list]
    # Context parameters
    kwargs = {}

    with tf.variable_scope("InputPipeline"):
        if mode == ModeKeys.TRAIN:
            if args.use_context and "hist" in features:
                kwargs["hist_noise"] = args.hist_noise
                kwargs["hist_noise_scale"] = args.hist_noise_scale
                kwargs["hist_scale"] = args.hist_scale
            return get_dataset_for_train(dataset,
                                         tumor_percent=TUMOR_PERCENT,
                                         random_scale=args.zoom_scale,
                                         context_guide=args.use_context,
                                         context_list=tuple(context_list),
                                         spatial_guide=args.use_spatial,
                                         spatial_random=args.spatial_random,
                                         spatial_inner_random=args.spatial_inner_random,
                                         config=args,
                                         **kwargs)
        elif mode == "eval_online":
            if args.use_context and "hist" in features:
                kwargs["hist_scale"] = args.hist_scale
            return get_dataset_for_eval_online(dataset,
                                               tumor_percent=TUMOR_PERCENT,
                                               context_guide=args.use_context,
                                               context_list=tuple(context_list),
                                               spatial_guide=args.use_spatial,
                                               spatial_random=0. if args.spatial_random < 1. else args.spatial_random,
                                               config=args,
                                               **kwargs)
        elif mode == ModeKeys.EVAL:
            if args.use_context and "hist" in features:
                kwargs["hist_scale"] = args.hist_scale
            return get_dataset_for_sp_point(dataset,
                                            config=args,
                                            context_guide=args.use_context,
                                            context_list=tuple(context_list),
                                            spatial_guide=args.use_spatial,
                                            **kwargs)

#####################################
#
#   G-Net input pipeline
#
#####################################


def data_processing_train(im_files, seg_file, read_size, bbox, PID_ci, img_clip, guide,
                          config, random_noise, random_flip_left_right, random_flip_up_down, **kwargs):
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
    use_context = "context" in guide
    use_spatial = "centers" in guide

    if use_context:
        features["context"] = guide["context"]

    def true_fn():
        stddevs = tf.maximum(guide["stddevs"], kwargs.get("min_std", 1.))
        gd = image_ops.create_spatial_guide_2d(guide["crop_size"], guide["centers"], stddevs)
        gd = tf.expand_dims(gd, axis=0)
        gd_resize = tf.image.resize_bilinear(gd, (config.im_height, config.im_width), align_corners=True)
        gd_resize = tf.squeeze(gd_resize, axis=0)
        return tf.cast(gd_resize / 2 + 0.5, tf.float32)

    def false_fn():
        return tf.ones((config.im_height, config.im_width, 1), tf.float32) * 0.5

    if use_spatial:
        features["sp_guide"] = tf.cond(tf.shape(guide["centers"])[0] > 0, true_fn, false_fn)

    if random_noise:
        features["images"] = image_ops.random_noise(features["images"], config.noise_scale, seed=RND_SEED)
        # Remove noise in empty slice
        features["images"] *= tf.cast(tf.greater(tf.strings.length(im_files), 0), tf.float32)
        logging.info("Train: Add random noise, scale = {}".format(config.noise_scale))
    if (random_flip_left_right or random_flip_up_down) and use_spatial:
        features["images"] = tf.concat((features["images"], features["sp_guide"]), axis=-1)
    if random_flip_left_right:
        features["images"], labels = image_ops.random_flip_left_right(features["images"], labels, RND_SEED)
        logging.info("Train: Add random flip left <-> right")
    if random_flip_up_down:
        features["images"], labels = image_ops.random_flip_up_down(features["images"], labels, RND_SEED)
        logging.info("Train: Add random flip up <-> down")
    if (random_flip_left_right or random_flip_up_down) and use_spatial:
        features["images"], features["sp_guide"] = tf.split(features["images"], [config.im_channel, 1], axis=-1)

    return features, labels


def gen_train_batch(data_list,
                    batch_size,
                    tumor_percent=0.,
                    random_scale=(1., 1.),
                    context_guide=False,
                    context_list=(("hist", 200),),
                    spatial_guide=False,
                    spatial_random=0.,
                    spatial_inner_random=False,
                    random_window_level=False,
                    config=None,
                    **kwargs):
    """ All coordinates are ij index """
    # Load context guide if needed
    context = {case["PID"]: None for case in data_list} if context_guide else None

    d = data_list
    list_of_keys = np.arange(len(d))
    tumor_list_of_keys = []
    for i in list_of_keys:
        if len(d[i]["slices"]) > 0:
            tumor_list_of_keys.append(i)
    target_size = np.asarray((config.im_height, config.im_width), dtype=np.float32)
    force_tumor = math.ceil(batch_size * tumor_percent)
    empty_mmts = np.zeros((0, 2), dtype=np.float32)

    while True:
        ci1 = np.random.choice(tumor_list_of_keys, force_tumor, True)
        ci2 = np.random.choice(list_of_keys, batch_size - force_tumor, True)  # case indices
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
                obj_bb = [size[1], size[2], 0, 0]  # Obj not exist
                if selected_slice in case["tumor_slices_index"]:
                    ind = case["tumor_slices_index"].index(selected_slice)
                else:
                    ind = -1

            # Compute crop region
            if obj_bb[2] - obj_bb[0] <= crop_size[0]:
                rng_yl = max(obj_bb[2] - crop_size[0], 0)
                rng_yr = min(obj_bb[0], size[1] - crop_size[0])
            else:
                rng_yl = max(obj_bb[0] + 20 - crop_size[0], 0)
                rng_yr = min(obj_bb[2] - 20, size[1] - crop_size[0])
            if obj_bb[3] - obj_bb[1] <= crop_size[1]:
                rng_xl = max(obj_bb[3] - crop_size[1], 0)
                rng_xr = min(obj_bb[1], size[2] - crop_size[1])
            else:
                rng_xl = max(obj_bb[1] + 20 - crop_size[1], 0)
                rng_xr = min(obj_bb[3] - 20, size[2] - crop_size[1])
            off_y = random.randint(rng_yl, rng_yr)
            off_x = random.randint(rng_xl, rng_xr)

            # Get multi-channel input
            selected_slices = [pattern.format(pid, selected_slice)]
            if config.im_channel > 1:
                left_half_channel = (config.im_channel - 1) // 2
                for k in range(1, left_half_channel + 1):
                    previous_slice = selected_slice - k
                    if 0 <= previous_slice < size[0]:
                        selected_slices.insert(0, pattern.format(pid, previous_slice))
                    else:
                        # slice index is "" means padding with zeros
                        selected_slices.insert(0, "")
                right_half_channel = config.im_channel - 1 - left_half_channel
                for k in range(1, right_half_channel + 1):
                    following_slice = selected_slice + k
                    if 0 <= following_slice < size[0]:
                        selected_slices.append(pattern.format(pid, following_slice))
                    else:
                        selected_slices.append("")

            # Random clip image value
            if random_window_level:
                img_clip = (0, random.randint(800, 1000))
            else:
                img_clip = (0, 900)

            # Last element store context guide and spatial guide information
            yield_list = (selected_slices, lb_pattern.format(pid, selected_slice), patch_size,
                          [off_y, off_x] + crop_size, pid, img_clip, {})

            use_guide = None, None
            if context_guide or spatial_guide:
                use_guide = random.random() < spatial_random
            # context guide
            if context_guide:
                # Load texture features when needed
                if context[pid] is None:
                    context[pid] = {}
                    gd_pattern = PROJ_ROOT / "data/NF/feat"
                    # We want context_mode choose from [train, eval], and else raise error
                    gd_pattern = str(gd_pattern / "%s" / kwargs.get("context_mode", None) / "%03d.npy")
                    for cls, f_len in context_list:
                        feat = np.load(gd_pattern % (cls, pid), allow_pickle=True)
                        assert isinstance(feat, np.ndarray), "`feat` must be a numpy.ndarray"
                        assert feat.shape[1] == f_len, "feature length mismatch %d vs %d" \
                                                       % (feat.shape[1], f_len)
                        context[pid][cls] = eval("feature_ops.%s_preprocess" % cls)(feat, **kwargs)

                if use_guide:
                    features = []   # Collect features of selected slice
                    for cls, _ in context_list:
                        if cls == "hist":
                            feat = context[pid]["hist"][selected_slice]
                            if "hist_noise" in kwargs and kwargs["hist_noise"]:
                                feat += np.random.normal(
                                    loc=0., scale=1., size=feat.shape) * kwargs.get("hist_noise_scale", 0.005)
                            features.append(feat)
                        elif cls == "glcm":
                            feat = context[pid]["glcm"][selected_slice]
                            if "glcm_noise" in kwargs:
                                # We use 1% value scale(between 2.5% percentile and 97.5% percentile)
                                # of each feature as random scale
                                feat += np.random.normal(
                                    loc=0., scale=1., size=feat.shape) * glcm_noise_scale
                            features.append(feat)
                        else:
                            features.append(context[pid][cls][selected_slice])
                    yield_list[-1]["context"] = np.concatenate(features, axis=0)
                else:
                    feat_length = sum([x for _, x in context_list])
                    yield_list[-1]["context"] = np.zeros((feat_length,), dtype=np.float32)

            # spatial guide
            if spatial_guide:
                if use_guide and ind >= 0:
                    centers = np.asarray(case["centers"][ind], dtype=np.float32)
                    # stddevs = np.asarray(case["stddevs"][ind], dtype=np.float32)
                    stddevs = np.array([[5., 5.]] * len(case["stddevs"][ind]), dtype=np.float32)
                    num_of_tumors = centers.shape[0]
                    tumor_choose = list(range(num_of_tumors))
                    inbox_tumors = []
                    for tc in tumor_choose:
                        if off_y <= centers[tc][0] < off_y + crop_size[0] and \
                                off_x <= centers[tc][1] < off_x + crop_size[1]:
                            inbox_tumors.append(tc)
                    if spatial_inner_random:
                        inbox_choose = random.sample(inbox_tumors, k=random.randint(1, len(inbox_tumors)))
                    else:
                        inbox_choose = inbox_tumors
                    new_c = centers[inbox_choose] - np.array([off_y, off_x])
                    new_s = stddevs[inbox_choose]
                    rand_c = new_s * np.random.uniform(-config.center_random_ratio,
                                                       config.center_random_ratio, new_c.shape) + new_c
                    rand_s = new_s * np.random.uniform(1. / (1 + config.stddev_random_ratio),
                                                       1. + config.stddev_random_ratio, new_s.shape)
                    rand_s = np.maximum(rand_s, config.min_std)
                    yield_list[-1].update({"centers": rand_c,
                                           "stddevs": rand_s,
                                           "crop_size": crop_size})
                else:
                    yield_list[-1].update({"centers": empty_mmts,
                                           "stddevs": empty_mmts,
                                           "crop_size": crop_size})
            yield yield_list


def get_dataset_for_train(data_list,
                          tumor_percent=0.,
                          random_scale=(1., 1.),
                          context_guide=False,
                          context_list=(),
                          spatial_guide=False,
                          spatial_random=0.,
                          spatial_inner_random=False,
                          config=None,
                          **kwargs):
    batch_size = distribution_utils.per_device_batch_size(config.batch_size, config.num_gpus)
    if random_scale[1] > random_scale[0]:
        logging.info("Train: Add random zoom, scale = ({}, {})".format(*random_scale))

    def train_gen():
        return gen_train_batch(data_list, batch_size, tumor_percent,
                               random_scale=random_scale,
                               context_guide=context_guide,
                               context_list=context_list,
                               spatial_guide=spatial_guide,
                               spatial_random=spatial_random,
                               spatial_inner_random=spatial_inner_random,
                               random_window_level=True,
                               config=config,
                               **kwargs)

    guide_types_dict = {}
    guide_shapes_dict = {}
    if context_guide:
        guide_types_dict["context"] = tf.float32
        feature_length = sum([x for _, x in context_list])
        guide_shapes_dict["context"] = tf.TensorShape([feature_length])
        logging.info("Train: Use context guide")
        if "hist_noise" in kwargs:
            logging.info("Train: Add context-hist noise, scale = {}".format(
                kwargs.get("hist_noise_scale", "Unknown")))
        kwargs["context_mode"] = "train"
    if spatial_guide:
        guide_types_dict.update({"centers": tf.float32,
                                 "stddevs": tf.float32,
                                 "crop_size": tf.int32})
        guide_shapes_dict.update({"centers": tf.TensorShape([None, 2]),
                                  "stddevs": tf.TensorShape([None, 2]),
                                  "crop_size": tf.TensorShape([2])})
        logging.info("Train: Use spatial guide")
        if spatial_random < 1:
            logging.info("Train: Add spatial guide random, prob = {}".format(spatial_random))
        if spatial_inner_random:
            logging.info("Train: Add spatial guide random in each slice")

    output_types = (tf.string, tf.string, tf.int32, tf.int32, tf.int32, tf.float32, guide_types_dict)
    output_shapes = (tf.TensorShape([config.im_channel]),
                     tf.TensorShape([]),
                     tf.TensorShape([3]),
                     tf.TensorShape([4]),
                     tf.TensorShape([]),
                     tf.TensorShape([2]),
                     guide_shapes_dict)

    dataset = (tf.data.Dataset.from_generator(train_gen, output_types, output_shapes)
               .apply(tf.data.experimental.map_and_batch(
                        lambda *args_: data_processing_train(*args_,
                                                             config=config,
                                                             random_noise=True,
                                                             random_flip_left_right=config.random_flip & 1 > 0,
                                                             random_flip_up_down=config.random_flip & 2 > 0,
                                                             mode="Train", **kwargs),
                        batch_size, num_parallel_batches=1))
               .prefetch(buffer_size=contrib_data.AUTOTUNE))

    return dataset


def get_dataset_for_eval_online(data_list,
                                tumor_percent=0.,
                                context_guide=False,
                                context_list=(),
                                spatial_guide=False,
                                spatial_random=0.,
                                config=None,
                                **kwargs):
    batch_size = distribution_utils.per_device_batch_size(config.batch_size, config.num_gpus)

    def eval_2d_gen():
        infinity_generator = gen_train_batch(data_list, batch_size, tumor_percent,
                                             random_scale=(1., 1.),
                                             context_guide=context_guide,
                                             context_list=context_list,
                                             spatial_guide=spatial_guide,
                                             spatial_random=spatial_random,
                                             spatial_inner_random=False,
                                             random_window_level=False,
                                             config=config,
                                             **kwargs)
        for _ in tqdm.tqdm(range(config.eval_num_batches_per_epoch * config.batch_size)):
            yield next(infinity_generator)

    guide_types_dict = {}
    guide_shapes_dict = {}
    if context_guide:
        guide_types_dict["context"] = tf.float32
        feature_length = sum([x for _, x in context_list])
        guide_shapes_dict["context"] = tf.TensorShape([feature_length])
        logging.info("Train: Use context guide")
        kwargs["context_mode"] = "eval"
    if spatial_guide:
        guide_types_dict.update({"centers": tf.float32,
                                 "stddevs": tf.float32,
                                 "crop_size": tf.int32})
        guide_shapes_dict.update({"centers": tf.TensorShape([None, 2]),
                                  "stddevs": tf.TensorShape([None, 2]),
                                  "crop_size": tf.TensorShape([2])})
        logging.info("Train: Use spatial guide")

    output_types = (tf.string, tf.string, tf.int32, tf.int32, tf.int32, tf.float32, guide_types_dict)
    output_shapes = (tf.TensorShape([config.im_channel]),
                     tf.TensorShape([]),
                     tf.TensorShape([3]),
                     tf.TensorShape([4]),
                     tf.TensorShape([]),
                     tf.TensorShape([2]),
                     guide_shapes_dict)

    dataset = (tf.data.Dataset.from_generator(eval_2d_gen, output_types, output_shapes)
               .apply(tf.data.experimental.map_and_batch(
                        lambda *args_: data_processing_train(*args_,
                                                             config=config,
                                                             random_noise=False,
                                                             random_flip_left_right=False,
                                                             random_flip_up_down=False,
                                                             mode="Eval Online"),
                        batch_size, num_parallel_batches=1))
               .prefetch(buffer_size=contrib_data.AUTOTUNE))

    return dataset


def parse_case_eval(case, im_channel, parse_label=True):
    """ Return cropped normalized volume (y, x, z) with type float32 and
               cropped segmentation (z, y, x) with type uint8 """
    d, h, w = case["size"]

    _, volume = nii_kits.read_nii(PROJ_ROOT / case["vol_case"])
    left_half_channel = (im_channel - 1) // 2
    right_half_channel = im_channel - 1 - left_half_channel
    cd, ch, cw = volume.shape
    volume = (np.clip(volume, GRAY_MIN, GRAY_MAX) - GRAY_MIN) / (GRAY_MAX - GRAY_MIN)
    volume = volume.transpose((1, 2, 0)).astype(np.float32)  # (y, x, z) for convenient

    segmentation = None
    lab_case = None
    if parse_label:
        _, segmentation = nii_kits.read_nii(PROJ_ROOT / case["lab_case"])
        segmentation = np.clip(segmentation.astype(np.uint8), 0, 1)
        lab_case = case["lab_case"]

    oshape = [d, h, w]
    cshape = [cd, ch, cw]
    return case["PID"], case["vol_case"], lab_case, oshape, cshape, \
        left_half_channel, right_half_channel, volume, segmentation


def get_dataset_for_sp_point(data_list, config, context_guide=None, context_list=(), spatial_guide=None, **kwargs):
    """ For spatial guide (only points) without context guide
        This function also support partial-guide strategy
    """
    batch_size = config.batch_size
    c = config.im_channel
    pshape = config.im_height, config.im_width
    gd_pattern = str(PROJ_ROOT / "data/NF/feat/%s/eval/%03d.npy")

    real_meta = None
    if config.real_sp and Path(config.real_sp).exists():
        logging.info("############# Load real spatial guide")
        with Path(config.real_sp).open() as f:
            real_meta = json.load(f)

    for ci, case in enumerate(data_list[config.eval_skip_num:]):
        pid, vol_path, _, oshape, cshape, lhc, rhc, volume, segmentation = parse_case_eval(case, c)
        spid = str(pid)

        pads = (batch_size - (cshape[0] % batch_size)) % batch_size
        volume = np.concatenate((np.zeros((*cshape[1:], lhc), volume.dtype),
                                 volume,
                                 np.zeros((*cshape[1:], pads + rhc), volume.dtype)), axis=-1)
        volume = cv2.resize(volume, pshape[::-1], interpolation=cv2.INTER_LINEAR)

        context_val = None
        if context_guide:
            # Load evaluation context
            features = []
            feat_length = 0
            for cls, f_len in context_list:
                feat = np.load(gd_pattern % (cls, pid), allow_pickle=True)
                assert isinstance(feat, np.ndarray), "`feat` must be a numpy.ndarray"
                assert feat.shape[1] == f_len, "feature length mismatch %d vs %d" % (feat.shape[1], f_len)
                features.append(eval("feature_ops.%s_preprocess" % cls)(feat, **kwargs))
                feat_length += f_len
            context_val = np.concatenate(features, axis=1).astype(np.float32)
            # Avoid index exceed array range
            context_val = np.concatenate((context_val, np.zeros((pads, feat_length), context_val.dtype)), axis=0)

        num_of_batches = (volume.shape[-1] - lhc - rhc) // batch_size
        assert volume.shape[-1] - lhc - rhc == batch_size * num_of_batches, \
            "Wrong padding: volume length: {}, lhc: {}, rhc: {}, batch_size: {}, num_of_batches: {}".format(
                volume.shape[-1], lhc, rhc, batch_size, num_of_batches)
        for idx in range(lhc, volume.shape[-1] - rhc, batch_size):
            sid = idx - lhc
            eval_batch = {"images": np.empty((batch_size, *pshape, c), dtype=np.float32),
                          "names": pid,
                          "mirror": 0}
            for j in range(batch_size):
                ssid = str(sid + j)
                eval_batch["images"][j] = volume[:, :, idx + j - lhc:idx + j + rhc + 1]
                if real_meta is not None and spid in real_meta and ssid in real_meta[spid]:
                    guide = array_kits.create_gaussian_distribution_v2(
                        cshape[1:], np.array(real_meta[spid][ssid]["centers"]), real_meta[spid][ssid]["stddevs"])
                    guide = guide * config.eval_discount / 2 + 0.5
                    guide = cv2.resize(guide, pshape[::-1], interpolation=cv2.INTER_LINEAR)
                    if "sp_guide" not in eval_batch:
                        eval_batch["sp_guide"] = np.ones((batch_size, *pshape, 1), dtype=np.float32) * 0.5
                    eval_batch["sp_guide"][j, ..., 0] = guide
                elif spatial_guide and not config.eval_no_sp and sid + j in case["tumor_slices_index"]:
                    ind = case["tumor_slices_index"].index(sid + j)
                    pos_centers = np.asarray(case["centers"][ind], dtype=np.float32)
                    pos_stddevs = np.array([[5., 5.]] * len(case["stddevs"][ind]), dtype=np.float32)
                    pos_centers = pos_centers / oshape[1:] * pshape
                    pos_stddevs = pos_stddevs / oshape[1:] * pshape
                    sp_guide = array_kits.create_gaussian_distribution_v2(pshape, pos_centers, pos_stddevs)

                    if "neg_centers" in case and sid + j in case["neg_tumor_slices_index"]:
                        nind = case["neg_tumor_slices_index"].index(sid + j)
                        neg_centers = np.asarray(case["neg_centers"][nind], dtype=np.float32)
                        neg_stddevs = np.array([[5., 5.]] * len(case["neg_stddevs"][nind]), dtype=np.float32)
                        neg_centers = neg_centers / oshape[1:] * pshape
                        neg_stddevs = neg_stddevs / oshape[1:] * pshape
                        neg_sp_guide = array_kits.create_gaussian_distribution_v2(pshape, neg_centers, neg_stddevs)
                        sp_guide -= neg_sp_guide * 2

                    sp_guide = sp_guide / 2 + 0.5
                    if "sp_guide" not in eval_batch:
                        eval_batch["sp_guide"] = np.ones((batch_size, *pshape, 1), dtype=np.float32) * 0.5
                    eval_batch["sp_guide"][j, ..., 0] = sp_guide
                if "sp_guide" not in eval_batch:
                    eval_batch["sp_guide"] = np.ones((batch_size, *pshape, 1), dtype=np.float32) * 0.5
                if context_guide:
                    eval_batch["context"] = context_val[sid:sid + batch_size]
            yield copy.copy(eval_batch), None

            if config.eval_mirror:
                if config.random_flip & 1 > 0:
                    tmp = copy.copy(eval_batch)
                    tmp["images"] = np.flip(tmp["images"], axis=2)
                    if spatial_guide:
                        tmp["sp_guide"] = np.flip(tmp["sp_guide"], axis=2)
                    tmp["mirror"] = 1
                    yield tmp, None
                if config.random_flip & 2 > 0:
                    tmp = copy.copy(eval_batch)
                    tmp["images"] = np.flip(tmp["images"], axis=1)
                    if spatial_guide:
                        tmp["sp_guide"] = np.flip(tmp["sp_guide"], axis=1)
                    tmp["mirror"] = 2
                    yield tmp, None
                if config.random_flip & 3 > 0:
                    tmp = copy.copy(eval_batch)
                    tmp["images"] = np.flip(np.flip(tmp["images"], axis=2), axis=1)
                    if spatial_guide:
                        tmp["sp_guide"] = np.flip(np.flip(tmp["sp_guide"], axis=2), axis=1)
                    tmp["mirror"] = 3
                    yield tmp, None
        yield None, (segmentation, vol_path, pads, oshape, True)


def get_dataset_for_eval_image(data_list, context_guide, context_list, spatial_guide, config=None, **kwargs):
    """ For context guide without spatial guide in evaluation
        or use spatial guide in training mode
    """

    batch_size = config.batch_size
    c = config.im_channel
    pshape = config.im_height, config.im_width
    gd_pattern = str(PROJ_ROOT / "data/NF/feat/%s/eval/%03d.npy")

    for ci, case in enumerate(data_list[config.eval_skip_num:]):
        pid, _, seg_path, oshape, cshape, lhc, rhc, volume, segmentation = parse_case_eval(case, c)

        # Load evaluation context
        features = []
        feat_length = 0
        for cls, f_len in context_list:
            feat = np.load(gd_pattern % (cls, pid), allow_pickle=True)
            assert isinstance(feat, np.ndarray), "`feat` must be a numpy.ndarray"
            assert feat.shape[1] == f_len, "feature length mismatch %d vs %d" % (feat.shape[1], f_len)
            features.append(eval("feature_ops.%s_preprocess" % cls)(feat, **kwargs))
            feat_length += f_len
        context_val = np.concatenate(features, axis=1).astype(np.float32)

        eval_batch = {"images": np.empty((batch_size, *pshape, c), dtype=np.float32),
                      "names": pid,
                      "context": np.zeros((batch_size, feat_length), dtype=np.float32),
                      "sp_guide": np.ones((batch_size, *pshape, 1), dtype=np.float32) * 0.5,
                      "mirror": 0}

        pads = (batch_size - (cshape[0] % batch_size)) % batch_size
        volume = np.concatenate((np.zeros((*cshape[1:], lhc), volume.dtype),
                                 volume,
                                 np.zeros((*cshape[1:], pads + rhc), volume.dtype)), axis=-1)
        volume = cv2.resize(volume, pshape[::-1], interpolation=cv2.INTER_LINEAR)
        # Avoid index exceed array range
        context_val = np.concatenate((context_val, np.zeros((pads, feat_length), context_val.dtype)), axis=0)

        num_of_batches = (volume.shape[-1] - lhc - rhc) // batch_size
        assert volume.shape[-1] - lhc - rhc == batch_size * num_of_batches, \
            "Wrong padding: volume length: {}, lhc: {}, rhc: {}, batch_size: {}, num_of_batches: {}".format(
                volume.shape[-1], lhc, rhc, batch_size, num_of_batches)
        for idx in range(lhc, volume.shape[-1] - rhc, batch_size):
            sid = idx - lhc

            for j in range(batch_size):
                eval_batch["images"][j] = volume[:, :, idx + j - lhc:idx + j + rhc + 1]
            if context_guide:
                eval_batch["context"] = context_val[sid:sid + batch_size]
            if spatial_guide:
                eval_batch["sp_guide"] = None
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
        yield None, (segmentation, seg_path, pads, oshape, True)


if __name__ == "__main__":
    # gen_dataset_jsons()
    pattern = str(PROJ_ROOT / "data/NF/png/volume-{:03d}/{:03d}_im.png")
    lb_pattern = str(PROJ_ROOT / "data/NF/png/volume-{:03d}/{:03d}_lb.png")
    # pattern = str(Path("D:/DataSet") / "LiTS/png/volume-{:d}/{:03d}_im.png")
    # lb_pattern = str(Path("D:/DataSet") / "LiTS/png/volume-{:d}/{:03d}_lb.png")
    import matplotlib
    matplotlib.use("Qt5Agg")
    import matplotlib.pyplot as plt

    np.random.seed(1234)
    tf.set_random_seed(1234)
    class Foo(object):
        batch_size = 8
        im_height = 256
        im_width = 256
        im_channel = 3
        num_gpus = 1
        noise_scale = 0.
        random_flip = 0
        center_random_ratio = 0.
        stddev_random_ratio = 0.
        min_std = 2.

    cnt = 0

    def show(f=None, l=None, idx=0):
        if f is not None:
            plt.subplot(231)
            plt.imshow(f["images"][idx, :, :, 0], cmap="gray")
            plt.subplot(232)
            plt.imshow(f["images"][idx, :, :, 1], cmap="gray")
            plt.subplot(233)
            plt.imshow(f["images"][idx, :, :, 2], cmap="gray")
        if "sp_guide" in f:
            plt.subplot(234)
            plt.imshow(f["sp_guide"][idx, :, :, 0], cmap="gray")
        if l is not None:
            plt.subplot(235)
            plt.imshow(l[idx, :, :], cmap="gray")
        plt.show()

    data_list = _get_datasets(choices=[35, 36])["choices"]
    d = get_dataset_for_train(data_list, TUMOR_PERCENT, RND_SCALE, False, (), True, 1.0, config=Foo())
    # lst = None
    # while True:
    #     try:
    #         f, l = next(gen)
    #         cnt, lst = save(f, l, cnt, lst)
    #     except StopIteration:
    #         break
    ff, ll = d.make_one_shot_iterator().get_next()
    sess = tf.Session()
    f, l = sess.run([ff, ll])
    show(f, l, 0)
