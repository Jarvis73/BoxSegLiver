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
pattern = str(PROJ_ROOT / "data/LiTS/png/volume-{:d}/{:03d}_im.png")
lb_pattern = str(PROJ_ROOT / "data/LiTS/png/volume-{:d}/{:03d}_lb.png")
GRAY_MIN = -200
GRAY_MAX = 250
IM_SCALE = 64
LB_SCALE = 64
LIVER_PERCENT = 0.66
TUMOR_PERCENT = 0.5
RND_SCALE = (1.0, 1.4)

RND_SEED = None     # For debug
# random.seed(1234)
# np.random.seed(1234)

# Pre-computed glcm noise scale
glcm_noise_scale = np.array([0.0004, 0.0008, 0.0005, 0.0008, 0.001 , 0.0008, 0.0012, 0.0008, 0.0013, 0.0014,
                             0.0015, 0.0014, 0.0013, 0.0016, 0.0013, 0.0017, 0.0019, 0.0016, 0.0021, 0.0017,
                             0.0021, 0.0021, 0.0023, 0.0022, 0.0045, 0.0034, 0.0041, 0.0034, 0.003 , 0.0034,
                             0.0028, 0.0034, 0.0025, 0.0025, 0.0025, 0.0025, 0.0019, 0.002 , 0.0019, 0.0021,
                             0.0021, 0.002 , 0.0021, 0.0021, 0.0023, 0.0025, 0.0023, 0.0025, 0.0043, 0.0046,
                             0.0043, 0.0046, 0.0048, 0.0046, 0.0048, 0.0046, 0.0051, 0.0053, 0.0051, 0.0052,
                             0.0038, 0.0067, 0.0045, 0.0069, 0.0087, 0.0067, 0.0093, 0.0069, 0.01  , 0.0108,
                             0.0106, 0.011 , 0.0262, 0.0248, 0.0262, 0.025 , 0.024 , 0.0248, 0.0243, 0.025,
                             0.0217, 0.0207, 0.022 , 0.022 , 0.1137, 0.1054, 0.1165, 0.1095, 0.1004, 0.1054,
                             0.1026, 0.1095, 0.094 , 0.0908, 0.0934, 0.0929], np.float32)


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
    # group.add_argument("--glcm_features", type=str, nargs="+",
    #                    choices=["contrast", "dissimilarity", "homogeneity", "ASM", "energy", "correlation"],
    #                    default=["contrast", "dissimilarity", "homogeneity", "energy", "correlation"],
    #                    help="Supported GLCM texture features")
    # group.add_argument("--glcm_distance", type=int, nargs="+", default=[1, 2, 3])
    # group.add_argument("--glcm_angle", type=float, nargs="+", default=[0., np.pi * 0.25, np.pi * 0.5, np.pi * 0.75])
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
    group.add_argument("--real_sp", type=str, help="Path to real spatial guide.")


def _get_datasets(test_fold=-1, filter_size=0, choices=None, exclude=None):
    prepare_dir = Path(__file__).parent / "prepare"
    prepare_dir.mkdir(parents=True, exist_ok=True)

    # Check existence
    obj_file = prepare_dir / ("dataset_f%d_fs%d.json" % (test_fold, filter_size))
    if obj_file.exists():
        with obj_file.open() as f:
            dataset_dict = json.load(f)
        return dataset_dict

    # Load meta.json
    meta = misc.load_meta("Liver", "LiTS/png")

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
        # Here we leave a postern for loading determined dataset (Using 3D-IRCAD-B as validation set)
        if test_fold == 73239:  # magic number
            print("Custom:: train/val split")
            print("         Train: 0~27, 48~130")
            print("         Val: 28~47 (3D-IRCAD-B)")
            trainset = list(range(28)) + list(range(48, 131))
            testset = list(range(28, 48))
        else:
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
                testset = []
            else:
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

    return {"infer": dataset_dict}


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
    if mode != ModeKeys.PREDICT:
        logging.info("{}: {} Liver CTs ({} slices, {} slices contain livers, {} slices contain tumors)"
                     .format(mode[:1].upper() + mode[1:], len(dataset),
                             sum([x["size"][0] for x in dataset]),
                             sum([x["bbox"][3] - x["bbox"][0] for x in dataset]),
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
                                         liver_percent=LIVER_PERCENT,
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
                                               liver_percent=LIVER_PERCENT,
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
            if args.eval_no_sp:
                return get_dataset_for_eval_image_sp(dataset, tuple(context_list), config=args)
            if args.use_spatial:
                return EvalImage3DLoader(dataset,
                                         context_guide=args.use_context,
                                         context_list=tuple(context_list),
                                         spatial_guide=args.use_spatial,
                                         config=args,
                                         **kwargs)
            elif args.use_context:
                return get_dataset_for_eval_image(dataset,
                                                  context_list=tuple(context_list),
                                                  config=args,
                                                  **kwargs)
            else:
                raise ValueError("Cannot set both use_context and use_spatial to false")
        elif mode == ModeKeys.PREDICT:
            if args.use_context and "hist" in features:
                kwargs["hist_scale"] = args.hist_scale
            if args.eval_no_sp or args.use_context:
                return get_dataset_for_infer(dataset,
                                             context_guide=args.use_context,
                                             context_list=tuple(context_list),
                                             config=args, **kwargs)
            if args.use_spatial:
                return EvalImage3DLoader(dataset,
                                         context_guide=args.use_context,
                                         context_list=tuple(context_list),
                                         spatial_guide=args.use_spatial,
                                         config=args,
                                         **kwargs)


#####################################
#
#   G-Net input pipeline
#
#####################################


def data_processing_train(im_files, seg_file, bbox, PID_ci, img_clip, guide, lab_scale,
                          config, random_noise, random_flip_left_right, random_flip_up_down,
                          mode="Train", **kwargs):
    off_x, off_y, height, width = bbox[0], bbox[1], bbox[2], bbox[3]

    def parse_im(name):
        return tf.cond(tf.greater(tf.strings.length(name), 0),
                       lambda: tf.image.decode_png(tf.io.read_file(name), channels=1, dtype=tf.uint16),
                       lambda: tf.zeros((512, 512, 1), dtype=tf.uint16))

    img = tf.map_fn(parse_im, im_files, dtype=tf.uint16)
    img = tf.image.crop_to_bounding_box(img, off_x, off_y, height, width)
    img = tf.image.resize_bilinear(img, (config.im_height, config.im_width), align_corners=True)
    img = tf.transpose(tf.squeeze(img, axis=-1), perm=(1, 2, 0))
    img = tf.cast(img, tf.float32)
    clip_min, clip_max = tf.split(img_clip, 2)
    img = (tf.clip_by_value(img, clip_min, clip_max) - clip_min) / (clip_max - clip_min)

    seg = tf.cond(tf.greater(tf.strings.length(seg_file), 0),
                  lambda: tf.image.decode_png(tf.io.read_file(seg_file), dtype=tf.uint8),
                  lambda: tf.zeros((512, 512, 1), dtype=tf.uint8))
    seg = tf.image.crop_to_bounding_box(seg, off_x, off_y, height, width)
    seg = tf.expand_dims(seg, axis=0)
    seg = tf.image.resize_nearest_neighbor(seg, (config.im_height, config.im_width), align_corners=True)
    seg = tf.cast(seg / lab_scale, tf.int32)
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
                    liver_percent=0.,
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
    force_liver = math.ceil(batch_size * liver_percent)
    force_tumor = math.ceil(batch_size * tumor_percent)
    empty_mmts = np.zeros((0, 2), dtype=np.float32)

    while True:
        ci1 = np.random.choice(tumor_list_of_keys, force_tumor, True)
        ci2 = np.random.choice(list_of_keys, batch_size - force_tumor, True)  # case indices
        ci = np.concatenate((ci1, ci2), axis=0)
        liver_counter = 0
        tumor_counter = 0
        for j, i in enumerate(ci):
            case = d[i]
            crop_size = (target_size * np.random.uniform(*random_scale, size=2)).astype(np.int32).tolist()
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
                # Record tumor slice indices for spatial guide=
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
            rng_yl = max(obj_bb[2] + 5 - crop_size[0], 0)
            rng_yr = min(obj_bb[0] - 5, size[1] - crop_size[0])
            if rng_yl + 20 < rng_yr:
                off_y = random.randint(rng_yl, rng_yr)
            else:
                # obj_bbox size exceeds crop_size or less than 20 pixels for random choices,
                # we will crop part of object
                off_y = random.randint(max(obj_bb[0] - 20, 0),
                                       min(int(obj_bb[0] * .75 + obj_bb[2] * .25), size[1] - crop_size[0]))
            rng_xl = max(obj_bb[3] + 5 - crop_size[1], 0)
            rng_xr = min(obj_bb[1] - 5, size[2] - crop_size[1])
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
                img_clip = (random.randint(10, 50) * IM_SCALE * 1., random.randint(500, 540) * IM_SCALE * 1.)
            else:
                img_clip = (50 * IM_SCALE * 1., 500 * IM_SCALE * 1.)

            # Last element store context guide and spatial guide information
            yield_list = (selected_slices, lb_pattern.format(pid, selected_slice),
                          [off_y, off_x] + crop_size, pid, img_clip, {})

            use_spatial_guide = None, None
            if context_guide or spatial_guide:
                use_spatial_guide = random.random() < spatial_random
            # context guide
            if context_guide:
                # Load texture features when needed
                if context[pid] is None:
                    context[pid] = {}
                    gd_pattern = PROJ_ROOT / "data/LiTS/feat"
                    # We want context_mode choose from [train, eval], and else raise error
                    gd_pattern = str(gd_pattern / "%s" / kwargs.get("context_mode", None) / "%03d.npy")
                    for cls, f_len in context_list:
                        feat = np.load(gd_pattern % (cls, pid), allow_pickle=True)
                        assert isinstance(feat, np.ndarray), "`feat` must be a numpy.ndarray"
                        assert feat.shape[1] == f_len, "feature length mismatch %d vs %d" \
                                                       % (feat.shape[1], f_len)
                        context[pid][cls] = eval("feature_ops.%s_preprocess" % cls)(feat, **kwargs)

                if use_spatial_guide:
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
                if use_spatial_guide and ind >= 0:
                    centers = np.asarray(case["centers"][ind], dtype=np.float32)
                    stddevs = np.asarray(case["stddevs"][ind], dtype=np.float32)
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


def gen_eval_3d_online_batch(data_list,
                             batch_size,
                             context_guide=False,
                             context_list=(("hist", 200),),
                             spatial_guide=False,
                             config=None,
                             **kwargs):
    """ All coordinates are ij index
    """
    # Load context guide if needed
    context = {case["PID"]: None for case in data_list} if context_guide else None
    empty_mmts = np.zeros((0, 2), dtype=np.float32)

    for case in data_list:
        z1, y1, x1, z2, y2, x2 = case["bbox"]
        crop_size = [y2 - y1, x2 - x1]
        size = case["size"]
        pid = case["PID"]

        pads = (batch_size - ((z2 - z1) % batch_size)) % batch_size
        # Make sure right_half_channel < 100
        for selected_slice in list(range(z1, z2)) + [-100] * pads:
            # Get selected slice
            if selected_slice in case["tumor_slices_index"]:
                ind = case["tumor_slices_index"].index(selected_slice)
            else:
                ind = -1

            # Get multi-channel input
            selected_slices = [pattern.format(pid, selected_slice) if selected_slice >= 0 else ""]
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

            img_clip = (50 * IM_SCALE * 1., 500 * IM_SCALE * 1.)

            # Last element store context guide and spatial guide information
            yield_list = (selected_slices,
                          lb_pattern.format(pid, selected_slice) if selected_slice >= 0 else "",
                          [y1, x1] + crop_size, pid, img_clip, {})

            # context guide
            if context_guide:
                # Load texture features when needed
                if context[pid] is None:
                    context[pid] = {}
                    gd_pattern = PROJ_ROOT / "data/LiTS/feat"
                    # We want context_mode choose from [train, eval], and else raise error
                    gd_pattern = str(gd_pattern / "%s" / kwargs.get("context_mode", None) / "%03d.npy")
                    for cls, f_len in context_list:
                        feat = np.load(gd_pattern % (cls, pid), allow_pickle=True)
                        assert isinstance(feat, np.ndarray), "`feat` must be a numpy.ndarray"
                        assert feat.shape[1] == f_len, "feature length mismatch %d vs %d" \
                                                       % (feat.shape[1], f_len)
                        context[pid][cls] = eval("feature_ops.%s_preprocess" % cls)(feat, **kwargs)

                features = []   # Collect features of selected slice
                for cls, _ in context_list:
                    if 0 <= selected_slice < size[0]:
                        features.append(context[pid][cls][selected_slice])
                    else:
                        features.append(np.zeros_like(context[pid][cls][0], dtype=context[pid][cls].dtype))
                yield_list[-1]["context"] = np.concatenate(features, axis=0)

            # spatial guide
            if spatial_guide:
                if ind >= 0:
                    centers = np.asarray(case["centers"][ind], dtype=np.float32)
                    stddevs = np.asarray(case["stddevs"][ind], dtype=np.float32)
                    centers -= np.array([y1, x1])
                    stddevs = np.maximum(stddevs, config.min_std)
                    yield_list[-1].update({"centers": centers,
                                           "stddevs": stddevs,
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
    logging.info("Train: Add random window level, scale = 50")

    def train_gen():
        return gen_train_batch(data_list, batch_size, liver_percent, tumor_percent,
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

    output_types = (tf.string, tf.string, tf.int32, tf.int32, tf.float32, guide_types_dict)
    output_shapes = (tf.TensorShape([config.im_channel]),
                     tf.TensorShape([]),
                     tf.TensorShape([4]),
                     tf.TensorShape([]),
                     tf.TensorShape([2]),
                     guide_shapes_dict)

    dataset = (tf.data.Dataset.from_generator(train_gen, output_types, output_shapes)
               .apply(tf.data.experimental.map_and_batch(
                        lambda *args_: data_processing_train(*args_,
                                                             lab_scale=LB_SCALE,
                                                             config=config,
                                                             random_noise=True,
                                                             random_flip_left_right=config.random_flip & 1 > 0,
                                                             random_flip_up_down=config.random_flip & 2 > 0,
                                                             mode="Train", **kwargs),
                        batch_size, num_parallel_batches=1))
               .prefetch(buffer_size=contrib_data.AUTOTUNE))

    return dataset


def get_dataset_for_eval_online(data_list,
                                liver_percent=0.,
                                tumor_percent=0.,
                                context_guide=False,
                                context_list=(),
                                spatial_guide=False,
                                spatial_random=0.,
                                config=None,
                                **kwargs):
    batch_size = distribution_utils.per_device_batch_size(config.batch_size, config.num_gpus)

    def eval_2d_gen():
        infinity_generator = gen_train_batch(data_list, batch_size, liver_percent, tumor_percent,
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

    def eval_3d_gen():
        val_generator = gen_eval_3d_online_batch(data_list, batch_size,
                                                 context_guide=context_guide,
                                                 context_list=context_list,
                                                 spatial_guide=spatial_guide,
                                                 config=config,
                                                 **kwargs)
        val_length = sum([(x["bbox"][3] - x["bbox"][0] + config.batch_size - 1) // config.batch_size
                          for x in data_list]) * config.batch_size
        for next_batch in tqdm.tqdm(val_generator, total=val_length):
            yield next_batch

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

    output_types = (tf.string, tf.string, tf.int32, tf.int32, tf.float32, guide_types_dict)
    output_shapes = (tf.TensorShape([config.im_channel]),
                     tf.TensorShape([]),
                     tf.TensorShape([4]),
                     tf.TensorShape([]),
                     tf.TensorShape([2]),
                     guide_shapes_dict)

    dataset = (tf.data.Dataset.from_generator(eval_3d_gen if config.eval_3d else eval_2d_gen,
                                              output_types, output_shapes)
               .apply(tf.data.experimental.map_and_batch(
                        lambda *args_: data_processing_train(*args_,
                                                             lab_scale=LB_SCALE,
                                                             config=config,
                                                             random_noise=False,
                                                             random_flip_left_right=False,
                                                             random_flip_up_down=False,
                                                             mode="Eval Online"),
                        batch_size, num_parallel_batches=1))
               .prefetch(buffer_size=contrib_data.AUTOTUNE))

    return dataset


def parse_case_eval(case, align, padding, padding_z, im_channel, parse_label=True, test_data=False):
    """ Return cropped normalized volume (y, x, z) with type float32 and
               cropped segmentation (z, y, x) with type uint8 """

    d, h, w = case["size"]
    z1 = max(case["bbox"][0] - padding_z, 0)
    z2 = min(case["bbox"][3] + padding_z, d)
    y1 = max(case["bbox"][1] - padding, 0)
    x1 = max(case["bbox"][2] - padding, 0)
    y2 = min(case["bbox"][4] + padding, h)
    x2 = min(case["bbox"][5] + padding, w)
    cy = (y1 + y2 - 1) / 2
    cx = (x1 + x2 - 1) / 2
    sz_y = int(math.ceil((y2 - y1) / align)) * align
    sz_x = int(math.ceil((x2 - x1) / align)) * align
    y1 = max(int(cy - (sz_y - 1) / 2), 0)
    x1 = max(int(cx - (sz_x - 1) / 2), 0)
    y2 = min(y1 + sz_y, h)
    x2 = min(x1 + sz_x, w)
    if (y2 - y1) % align != 0 or (x2 - x1) % align != 0:
        y1 = y2 - sz_y
        x1 = x2 - sz_x
        if y1 < 0 or x1 < 0:
            print("\nWarning: bbox aligns with {} failed! point1 ({}, {}) point2 ({}, {})\n"
                  .format(align, x1, y1, x2, y2))

    obj_num = int(case["vol_case"][:-4].split("-")[-1])
    if test_data:
        _, volume = nii_kits.read_nii(PROJ_ROOT / case["vol_case"])
    else:
        _, volume = nii_kits.read_lits(obj_num, "vol", PROJ_ROOT / case["vol_case"])
    left_half_channel = (im_channel - 1) // 2
    right_half_channel = im_channel - 1 - left_half_channel
    left_pad = left_half_channel - z1 if z1 < left_half_channel else 0
    right_pad = z2 + right_half_channel - d if z2 + right_half_channel > d else 0
    crop_z1 = max(0, z1 - left_half_channel)
    crop_z2 = min(d, z2 + right_half_channel)
    volume = volume[crop_z1:crop_z2, y1:y2, x1:x2]
    cd, ch, cw = volume.shape  # cd: cropped depth
    if left_pad > 0 or right_pad > 0:
        volume = np.concatenate((np.zeros((left_pad, ch, cw), dtype=volume.dtype),
                                 volume,
                                 np.zeros((right_pad, ch, cw), dtype=volume.dtype)), axis=0)
        cd, ch, cw = volume.shape
    volume = (np.clip(volume, GRAY_MIN, GRAY_MAX) - GRAY_MIN) / (GRAY_MAX - GRAY_MIN)
    volume = volume.transpose((1, 2, 0)).astype(np.float32)  # (y, x, z) for convenient

    segmentation = None
    lab_case = None
    if parse_label:
        _, segmentation = nii_kits.read_lits(obj_num, "lab", PROJ_ROOT / case["lab_case"])
        segmentation = segmentation.astype(np.uint8)[z1:z2, y1:y2, x1:x2]
        lab_case = case["lab_case"]

    bbox = [x1, y1, z1, x2 - 1, y2 - 1, z2 - 1]
    oshape = [d, h, w]
    cshape = [cd, ch, cw]
    return case["PID"], case["vol_case"], lab_case, bbox, oshape, cshape, \
        left_half_channel, right_half_channel, volume, segmentation


def get_dataset_for_eval_image(data_list, context_list, config=None, **kwargs):
    """ For context guide without spatial guide in evaluation
        or use spatial guide in training mode
    """
    align = 16
    padding = 25
    padding_z = 0

    batch_size = config.batch_size
    c = config.im_channel
    pshape = config.im_height, config.im_width
    gd_pattern = str(PROJ_ROOT / "data/LiTS/feat/%s/eval/%03d.npy")

    for ci, case in enumerate(data_list[config.eval_skip_num:]):
        pid, vol_path, _, bbox, oshape, cshape, lhc, rhc, volume, segmentation = \
            parse_case_eval(case, align, padding, padding_z, c)

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
                      "context": None,
                      "mirror": 0,
                      "direction": "Forward"}

        pads = (batch_size - ((bbox[5] - bbox[2] + 1) % batch_size)) % batch_size
        if pads > 0:
            volume = np.concatenate((volume, np.zeros((*cshape[1:], pads), volume.dtype)), axis=-1)
            # Avoid index exceed array range
            context_val = np.concatenate((context_val, np.zeros((pads, feat_length), context_val.dtype)), axis=0)
        volume = cv2.resize(volume, pshape, interpolation=cv2.INTER_LINEAR)

        num_of_batches = (volume.shape[-1] - lhc - rhc) // batch_size
        assert volume.shape[-1] - lhc - rhc == batch_size * num_of_batches, \
            "Wrong padding: volume length: {}, lhc: {}, rhc: {}, batch_size: {}, num_of_batches: {}".format(
                volume.shape[-1], lhc, rhc, batch_size, num_of_batches)
        for idx in range(lhc, volume.shape[-1] - rhc, batch_size):
            sid = bbox[2] + idx - lhc

            for j in range(batch_size):
                eval_batch["images"][j] = volume[:, :, idx + j - lhc:idx + j + rhc + 1]
            eval_batch["context"] = context_val[sid:sid + batch_size]
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
        yield None, (segmentation, vol_path, pads, bbox)


def get_dataset_for_eval_image_sp(data_list, context_list, config=None):
    """ For spatial guide without spatial guide in evaluation
    """
    align = 16
    padding = 25
    padding_z = 0

    batch_size = config.batch_size
    c = config.im_channel
    pshape = config.im_height, config.im_width
    feat_length = sum([x for _, x in context_list])

    real_meta = None
    if config.real_sp and Path(config.real_sp).exists():
        with Path(config.real_sp).open() as f:
            real_meta = json.load(f)

    for ci, case in enumerate(data_list[config.eval_skip_num:]):
        pid, vol_path, _, bbox, oshape, cshape, lhc, rhc, volume, segmentation = \
            parse_case_eval(case, align, padding, padding_z, c)
        spid = str(pid)

        eval_batch = {"images": np.empty((batch_size, *pshape, c), dtype=np.float32),
                      "names": pid,
                      "sp_guide": np.ones((batch_size, *pshape, 1), dtype=np.float32) * 0.5,
                      "context": np.zeros((batch_size, feat_length), dtype=np.float32),
                      "mirror": 0,
                      "direction": "Forward"}

        pads = (batch_size - ((bbox[5] - bbox[2] + 1) % batch_size)) % batch_size
        if pads > 0:
            volume = np.concatenate((volume, np.zeros((*cshape[1:], pads), volume.dtype)), axis=-1)
            # Avoid index exceed array range
        volume = cv2.resize(volume, pshape, interpolation=cv2.INTER_LINEAR)

        num_of_batches = (volume.shape[-1] - lhc - rhc) // batch_size
        assert volume.shape[-1] - lhc - rhc == batch_size * num_of_batches, \
            "Wrong padding: volume length: {}, lhc: {}, rhc: {}, batch_size: {}, num_of_batches: {}".format(
                volume.shape[-1], lhc, rhc, batch_size, num_of_batches)
        for idx in range(lhc, volume.shape[-1] - rhc, batch_size):
            ssid = str(bbox[2] + idx - lhc)
            has_guide = False
            for j in range(batch_size):
                eval_batch["images"][j] = volume[:, :, idx + j - lhc:idx + j + rhc + 1]
                if real_meta is not None and spid in real_meta and ssid in real_meta[spid]:
                    guide = array_kits.create_gaussian_distribution_v2(
                        cshape[1:], np.array(real_meta[spid][ssid]["centers"]) - bbox[1::-1],
                        real_meta[spid][ssid]["stddevs"])
                    guide = guide * config.eval_discount / 2 + 0.5
                    guide = cv2.resize(guide, pshape, interpolation=cv2.INTER_LINEAR)
                    eval_batch["sp_guide"][j] = guide[:, :, None]
                    has_guide = True

            yield copy.copy(eval_batch), None

            if config.eval_mirror:
                if config.random_flip & 1 > 0:
                    tmp = copy.copy(eval_batch)
                    tmp["images"] = np.flip(tmp["images"], axis=2)
                    if has_guide:
                        tmp["sp_guide"] = np.flip(tmp["sp_guide"], axis=2)
                    tmp["mirror"] = 1
                    yield tmp, None
                if config.random_flip & 2 > 0:
                    tmp = copy.copy(eval_batch)
                    tmp["images"] = np.flip(tmp["images"], axis=1)
                    if has_guide:
                        tmp["sp_guide"] = np.flip(tmp["sp_guide"], axis=1)
                    tmp["mirror"] = 2
                    yield tmp, None
                if config.random_flip & 3 > 0:
                    tmp = copy.copy(eval_batch)
                    tmp["images"] = np.flip(np.flip(tmp["images"], axis=2), axis=1)
                    if has_guide:
                        tmp["sp_guide"] = np.flip(np.flip(tmp["sp_guide"], axis=2), axis=1)
                    tmp["mirror"] = 3
                    yield tmp, None
        yield None, (segmentation, vol_path, pads, bbox, True)


def get_dataset_for_infer(data_list,
                          context_guide=None,
                          context_list=None,
                          config=None, **kwargs):
    """ For spatial guide without spatial guide in evaluation
    """
    align = 16
    padding = 25
    padding_z = 0

    batch_size = config.batch_size
    c = config.im_channel
    pshape = config.im_height, config.im_width
    resize = True
    logging.info("{} {} cases ...".format(config.mode.capitalize(), len(data_list)))
    # if config.im_height <= 0 or config.im_width <= 0:
    #     logging.info("Disable image resize for evaluating")
    #     resize = False

    real_meta = None
    if config.real_sp and Path(config.real_sp).exists():
        with Path(config.real_sp).open() as f:
            real_meta = json.load(f)

    context_val = None
    gd_pattern = str(PROJ_ROOT / "data/LiTS/feat/%s/infer/%03d.npy")

    for ci, case in enumerate(data_list[config.eval_skip_num:]):
        pid, vol_path, _, bbox, oshape, cshape, lhc, rhc, volume, segmentation = \
            parse_case_eval(case, align, padding, padding_z, c,
                            parse_label=config.mode != ModeKeys.PREDICT, test_data=True)
        if not resize:
            pshape = cshape[1:]
        spid = str(pid)

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

        eval_batch = {"images": np.empty((batch_size, *pshape, c), dtype=np.float32),
                      "names": pid,
                      "sp_guide": np.ones((batch_size, *pshape, 1), dtype=np.float32) * 0.5,
                      "context": None,
                      "mirror": 0,
                      "direction": "Forward"}

        pads = (batch_size - ((bbox[5] - bbox[2] + 1) % batch_size)) % batch_size
        if pads > 0:
            volume = np.concatenate((volume, np.zeros((*cshape[1:], pads), volume.dtype)), axis=-1)
            # Avoid index exceed array range
        d = volume.shape[-1]
        if resize:
            if d > 512:
                # cv2.resize can only resize at most 512 channels, so we must resize in batches
                volumes = []
                for b in range(0, d, 512):
                    volumes.append(cv2.resize(volume[..., b:b + 512], pshape, interpolation=cv2.INTER_LINEAR))
                volume = np.concatenate(volumes, axis=-1)
            else:
                volume = cv2.resize(volume, pshape, interpolation=cv2.INTER_LINEAR)

        num_of_batches = (volume.shape[-1] - lhc - rhc) // batch_size
        assert volume.shape[-1] - lhc - rhc == batch_size * num_of_batches, \
            "Wrong padding: volume length: {}, lhc: {}, rhc: {}, batch_size: {}, num_of_batches: {}".format(
                volume.shape[-1], lhc, rhc, batch_size, num_of_batches)
        for idx in range(lhc, volume.shape[-1] - rhc, batch_size):
            sid = bbox[2] + idx - lhc
            ssid = str(sid)
            has_guide = False
            for j in range(batch_size):
                eval_batch["images"][j] = volume[:, :, idx + j - lhc:idx + j + rhc + 1]
                if real_meta is not None and spid in real_meta and ssid in real_meta[spid]:
                    guide = array_kits.create_gaussian_distribution_v2(
                        cshape[1:], np.array(real_meta[spid][ssid]["centers"]) - bbox[1::-1],
                        real_meta[spid][ssid]["stddevs"])
                    guide = guide * config.eval_discount / 2 + 0.5
                    guide = cv2.resize(guide, pshape, interpolation=cv2.INTER_LINEAR)
                    eval_batch["sp_guide"][j] = guide[:, :, None]
                    has_guide = True
            if context_guide:
                eval_batch["context"] = context_val[sid:sid + batch_size]

            yield copy.copy(eval_batch), None

            if config.eval_mirror:
                if config.random_flip & 1 > 0:
                    tmp = copy.copy(eval_batch)
                    tmp["images"] = np.flip(tmp["images"], axis=2)
                    if has_guide:
                        tmp["sp_guide"] = np.flip(tmp["sp_guide"], axis=2)
                    tmp["mirror"] = 1
                    yield tmp, None
                if config.random_flip & 2 > 0:
                    tmp = copy.copy(eval_batch)
                    tmp["images"] = np.flip(tmp["images"], axis=1)
                    if has_guide:
                        tmp["sp_guide"] = np.flip(tmp["sp_guide"], axis=1)
                    tmp["mirror"] = 2
                    yield tmp, None
                if config.random_flip & 3 > 0:
                    tmp = copy.copy(eval_batch)
                    tmp["images"] = np.flip(np.flip(tmp["images"], axis=2), axis=1)
                    if has_guide:
                        tmp["sp_guide"] = np.flip(np.flip(tmp["sp_guide"], axis=2), axis=1)
                    tmp["mirror"] = 3
                    yield tmp, None
        yield None, (segmentation, vol_path, pads, bbox, resize)


class EvalImage3DLoader(object):
    """ For spatial guide propagation in evaluation """

    def __init__(self, data_list,
                 context_guide=False,
                 context_list=(),
                 spatial_guide=False,
                 config=None,
                 **kwargs):
        self.cfg = config
        self.data_list = data_list[self.cfg.eval_skip_num:]
        self.num_cases = self.cfg.eval_num if self.cfg.eval_num > 0 else len(self.data_list)
        logging.info("{} {} cases ...".format(self.cfg.mode.capitalize(), self.num_cases))
        self.cur_case_idx = -1
        self.use_context = context_guide
        self.context_list = context_list
        self.use_spatial = spatial_guide
        self.kwargs = kwargs
        self._last_guide = None     # [ph, pw]
        self._last_pred = None      # [ph, pw]
        self.min_std = self.cfg.min_std
        self.mirror_counter = 0
        self.sid = None
        self.case_iter = None
        self.direction = "Forward"
        self.oshape = None      # origin shape
        self.cshape = None      # cropped shape
        self.pshape = (self.cfg.im_height, self.cfg.im_width, self.cfg.im_channel)
        self.context_val = None
        self.labels = None
        self.last_info = []     # tracking tumors(come from last slice)
        self.curr_info = []     # current tumors
        self.sp_guides = []
        self.sp_guide_bg = 0.5
        self.filter_thresh = 0.15 + self.sp_guide_bg
        self.disc = ndi.generate_binary_structure(2, connectivity=1)
        self.gd_pattern = str(PROJ_ROOT / "data/LiTS/feat/%s/eval/%03d.npy")

        if not self.cfg.real_sp:
            if self.cfg.mode == ModeKeys.PREDICT:
                raise ValueError("`--real_sp` should be set when running inference.")
            prior_file = Path(__file__).parent / "prepare" / "prior.json"
        else:
            prior_file = Path(self.cfg.real_sp)
        with prior_file.open() as f:
            self.user_info = json.load(f)
        logging.info("Load prior file " + str(prior_file))

        self.debug = False

    def reset_counter(self):
        self.mirror_counter = 0

    def inc_counter(self):
        self.mirror_counter += 1

    @property
    def last_pred(self):
        return self._last_pred

    def print(self, *args_, **kwargs):
        if self.debug:
            print(*args_, **kwargs)

    def forward(self):
        return self.direction == "Forward"

    def backward(self):
        return self.direction == "Backward"

    @last_pred.setter
    def last_pred(self, new_pred):
        """
        Remove those predicted tumors who are out of range.
        Apply supervisor to predicted tumors.

        Make sure `pred` is binary

        self.pred which is created in this function will be used for generating next guide.
        So don't use `self.pred` as real prediction, because we will also remove those ended
        in current slice from self.pred.
        """
        if new_pred is None:
            return
        if self._last_guide is None:
            raise ValueError("previous_guide is None")
        new_pred = np.squeeze(new_pred, axis=(0, -1))
        if np.max(new_pred) == 0:
            self._last_pred = None
            return
        new_pred = new_pred.copy()
        self.last_info.clear()

        labeled_objs, n_objs = ndi.label(new_pred, self.disc)
        slicers = ndi.find_objects(labeled_objs)
        # Decide whether reaching the end of the tumor or not
        for i, slicer in zip(range(n_objs), slicers):
            res_obj = labeled_objs == i + 1
            res_obj_slicer = res_obj[slicer]
            # 1. For test: Filter wrong tumors(no corresponding guide)
            #    For infer: Filter not guided tumors(maybe some automatic predicted tumors)
            mask_guide_by_res = res_obj_slicer * self._last_guide[slicer]
            if np.max(mask_guide_by_res) < self.filter_thresh:
                self.print(self.sid, "Remove",
                           "Thresh:", self.filter_thresh,
                           "Max:", np.max(mask_guide_by_res),
                           "All last guide max:", np.max(self._last_guide))
                new_pred[slicer] -= res_obj_slicer
                continue
            # 2. Match res_obj to guide
            res_peak_pos = np.asarray(np.unravel_index(mask_guide_by_res.argmax(), mask_guide_by_res.shape))
            res_peak_pos[0] += slicer[0].start
            res_peak_pos[1] += slicer[1].start
            #   2.1. Check whether res_peak is just a guide center
            found = -1
            for j, obj in enumerate(self.curr_info):
                if np.all(res_peak_pos == obj["center"]):
                    found = j  # res_peak is just a center
                    break
            #   2.2. From the nearest guide center, check that whether it is the corresponding guide.
            #        Rule: Image(guide) values along the line from res_obj's peak to its corresponding
            #        guide center must be monotonously increasing.
            if found < 0:  # gradient ascent from res_peak to center
                # compute distances between res_obj_peak and every guide center
                distances = np.sum([(res_peak_pos - obj["center"]) ** 2 for obj in self.curr_info], axis=1)
                self.print(self.sid, "Distance:", distances)
                order = np.argsort(distances)
                for j in order:
                    ctr = self.curr_info[j]["center"]
                    self.print(self.sid, res_peak_pos[1], res_peak_pos[0], ctr[1], ctr[0])
                    self.print(self._last_guide[ctr[0]:ctr[0] + 3, ctr[1]:ctr[1] + 3])
                    if self.ascent_line(self._last_guide, res_peak_pos[1], res_peak_pos[0], ctr[1], ctr[0]):
                        # Found
                        found = j
                        break
            if found < 0:
                self.print(self.sid, "Res peak:", res_peak_pos, "Slice info:", self.curr_info)
                raise ValueError("Can not find corresponding guide!")
            # 3. Check z range and remove finished tumors(remove from next guide image)
            if (self.forward and self.sid >= self.curr_info[found]["z"][1]) or \
                    (self.backward and self.sid <= self.curr_info[found]["z"][0]):
                new_pred[slicer] -= res_obj_slicer
                continue
            # 4. Compute moments. Save moments of tumors for next slice
            ctr, std = array_kits.compute_robust_moments(res_obj_slicer, indexing="ij",
                                                         min_std=self.min_std)
            ctr[0] += slicer[0].start
            ctr[1] += slicer[1].start
            self.last_info.append({"z": copy.copy(self.curr_info[found]["z"]),
                                   "center": ctr.astype(np.int32).tolist(),
                                   "stddev": std.tolist()})
        self._last_pred = new_pred

    def prepare_next_case(self):
        self.cur_case_idx += 1
        if self.cur_case_idx >= self.num_cases:
            return False
        # Shape
        #   self.bbox: [x1, y1, z1, x2 - 1, y2 - 1, z2 - 1]
        #   self.oshape: [d, h, w]
        #   self.cshape: [cd, ch, cw] == self.volume.shape
        #   self.lhc: int
        #   self.rhc: int
        #   self.volume: [ch, cw, cd]
        #   self.segmentation: [cd, ch, cw]
        self.pid, self.vol_path, _, self.bbox, self.oshape, self.cshape, \
            self.lhc, self.rhc, self.volume, self.segmentation = \
            parse_case_eval(self.data_list[self.cur_case_idx],
                            align=16, padding=25, padding_z=0, im_channel=self.cfg.im_channel,
                            parse_label=self.cfg.mode != ModeKeys.PREDICT,
                            test_data=self.cfg.mode == ModeKeys.PREDICT)
        self.volume = cv2.resize(self.volume, self.pshape[:2], interpolation=cv2.INTER_LINEAR)[None]
        self.spid = str(self.pid)
        feat_length = 0
        if self.use_context:
            features = []
            for cls, f_len in self.context_list:
                feat = np.load(self.gd_pattern % (cls, self.pid), allow_pickle=True)
                assert isinstance(feat, np.ndarray), "`feat` must be a numpy.ndarray"
                assert feat.shape[1] == f_len, "feature length mismatch %d vs %d" \
                                               % (feat.shape[1], f_len)
                features.append(eval("feature_ops.%s_preprocess" % cls)(feat, **self.kwargs))
                feat_length += f_len
            self.context_val = np.concatenate(features, axis=1).astype(np.float32)
        self.case_iter = self.gen_next_batch()
        self.labels = (self.segmentation, self.vol_path, 0, self.bbox)
        return True

    def slice_iter(self, eval_batch, idx):
        # For bagging method on each slice
        for features in self.process_slice(eval_batch, idx):
            yield features
            if self.cfg.save_sp_guide and features["mirror"] == 0:
                self.sp_guides.append(eval_batch["sp_guide"])

    def gen_next_batch(self):
        self.direction = "Forward"
        eval_batch = {"images": None, "names": self.pid, "mirror": 0,
                      "context": None, "sp_guide": None, "direction": self.direction}
        for idx in range(self.lhc, self.cshape[0] - self.rhc):
            if idx == self.cshape[0] - self.rhc - 1:
                eval_batch["bboxes"] = self.bbox
            yield self.slice_iter(eval_batch, idx)
        if self.cfg.save_sp_guide:
            self._save_guide()
            self.sp_guides.clear()

        self.direction = "Backward"
        eval_batch = {"images": None, "names": self.pid, "mirror": 0,
                      "context": None, "sp_guide": None, "direction": self.direction}
        for idx in range(self.cshape[0] - self.rhc - 1, self.lhc - 1, -1):
            if idx == self.lhc:
                eval_batch["bboxes"] = self.bbox
            yield self.slice_iter(eval_batch, idx)
        if self.cfg.save_sp_guide:
            self._save_guide()
            self.sp_guides.clear()

    def process_slice(self, eval_batch, idx):
        zz1 = idx - self.lhc
        self.sid = zz1 + self.bbox[2]
        self.ssid = str(self.sid)    # string slice id
        eval_batch["images"] = self.volume[..., zz1:zz1 + self.pshape[-1]]
        # Add context guide
        if self.use_context:
            eval_batch["context"] = self.context_val[self.sid:self.sid + 1]

        # Add spatial guide
        if self.use_spatial:
            self.curr_info.clear()
            if self.ssid in self.user_info[self.spid]:
                self.print(self.sid, "New user info")
                this_info = copy.deepcopy(self.user_info[self.spid][self.ssid])
                # Here we must transform user_info centers(whose original point locate at (0, 0) of image)
                # to cropped image(whose original point locate at (bb[1], bb[0])). Then we zoom them to
                # patch size
                #
                #     (0, 0)*-------------------+
                #           | (bb[1], bb[0])    |
                #           |   *------------   |
                #           |   |           |   |
                #           |   |   o       |   |
                #           |   |           |   |
                #           |   |           |   |
                #           |   |           |   |
                #           |   +-----------+   |
                #           |                   |
                #           +-------------------+
                #
                for x in this_info:
                    # Filter stddev < min_std
                    if np.min(x["stddev"]) > self.min_std:
                        x["center"][0] = int((x["center"][0] - self.bbox[1]) / self.cshape[1] * self.pshape[0])
                        x["center"][1] = int((x["center"][1] - self.bbox[0]) / self.cshape[2] * self.pshape[1])
                        self.curr_info.append(x)

            # TODO(zjw): Maybe we should remove the same objects from centers_ys and centers_backup.
            #            For example, two or more adjacent slices are provided guides.
            if len(self.last_info) > 0:
                self.print(self.sid, "Last pred propagate:", self.last_info)
            self.curr_info.extend(self.last_info)
            centers = [x["center"] for x in self.curr_info]
            stddevs = [x["stddev"] for x in self.curr_info]
            if len(stddevs) > 0:
                assert np.min(stddevs) >= self.min_std, stddevs
                # guide has shape [ph, pw, 1]
                guide = array_kits.create_gaussian_distribution_v2(self.pshape[:2], centers, stddevs) * \
                    self.cfg.eval_discount
                self._last_guide = guide / 2 + self.sp_guide_bg
            else:
                self._last_guide = np.empty(self.pshape[:2], dtype=np.float32)
                self._last_guide.fill(self.sp_guide_bg)
            self.print(self.sid, "Current guide max:", np.max(self._last_guide))
            eval_batch["sp_guide"] = self._last_guide[None, :, :, None]
        yield copy.copy(eval_batch)

        if self.cfg.eval_mirror:
            if self.cfg.random_flip & 1 > 0:
                tmp = copy.copy(eval_batch)
                tmp["images"] = np.flip(tmp["images"], axis=2)
                if self.use_spatial:
                    tmp["sp_guide"] = np.flip(tmp["sp_guide"], axis=2)
                tmp["mirror"] = 1
                yield tmp
            if self.cfg.random_flip & 2 > 0:
                tmp = copy.copy(eval_batch)
                tmp["images"] = np.flip(tmp["images"], axis=1)
                if self.use_spatial:
                    tmp["sp_guide"] = np.flip(tmp["sp_guide"], axis=1)
                tmp["mirror"] = 2
                yield tmp
            if self.cfg.random_flip & 3 > 0:
                tmp = copy.copy(eval_batch)
                tmp["images"] = np.flip(np.flip(tmp["images"], axis=2), axis=1)
                if self.use_spatial:
                    tmp["sp_guide"] = np.flip(np.flip(tmp["sp_guide"], axis=2), axis=1)
                tmp["mirror"] = 3
                yield tmp

    def _save_guide(self):
        img_array = np.squeeze(np.concatenate(self.sp_guides, axis=0), axis=-1)
        # Resize logits3d to the shape of labels3d
        ori_shape = list(array_kits.bbox_to_shape(self.bbox))
        cur_shape = img_array.shape
        ori_shape[0] = cur_shape[0]
        scales = np.array(ori_shape) / np.array(cur_shape)
        img_array = ndi.zoom(img_array, scales, order=1)
        img_array = (img_array * 255).astype(np.int16)

        case_name = "guide-{}-{}.nii.gz".format(self.direction[0], self.pid)
        save_path = Path(self.cfg.model_dir) / "sp_guide"
        save_path.mkdir(parents=True, exist_ok=True)
        save_path = save_path / case_name

        header = nii_kits.read_lits(self.pid, "vol", self.vol_path, only_header=True)
        pad_with = tuple(zip(self.bbox[2::-1], np.array(header.get_data_shape()[::-1]) - self.bbox[:2:-1] - 1))
        img_array = np.pad(img_array, pad_with, mode="constant", constant_values=np.int16(self.sp_guide_bg * 255))
        nii_kits.write_nii(img_array, header, save_path, special=True if 28 <= self.pid < 48 else False)

    @staticmethod
    def ascent_line(img, x0, y0, x1, y1):
        # Find points along this line
        xs, ys, forward = array_kits.xiaolinwu_line(x0, y0, x1, y1)
        ascent = True
        pre = img[ys[0], xs[0]] if forward else img[ys[-1], xs[-1]]
        xs, ys = (xs, ys) if forward else (reversed(xs[:-1]), reversed(ys[:-1]))
        for x, y in zip(xs, ys):
            cur = img[y, x]
            if cur >= pre:
                pre = cur
                continue
            else:
                ascent = False
                break
        return ascent


if __name__ == "__main__":
    # gen_dataset_jsons()
    pattern = str(PROJ_ROOT / "data/LiTS/png/volume-{:d}/{:03d}_im.png")
    lb_pattern = str(PROJ_ROOT / "data/LiTS/png/volume-{:d}/{:03d}_lb.png")
    # pattern = str(Path("D:/DataSet") / "LiTS/png/volume-{:d}/{:03d}_im.png")
    # lb_pattern = str(Path("D:/DataSet") / "LiTS/png/volume-{:d}/{:03d}_lb.png")
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
        random_flip = 3
        center_random_ratio = 0.2
        stddev_random_ratio = 0.4
        eval_3d = True

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
        if "sp_guide" in f:
            plt.subplot(234)
            plt.imshow(f["sp_guide"][idx, :, :, 0], cmap="gray")
        if l is not None:
            plt.subplot(235)
            plt.imshow(l[idx, :, :], cmap="gray")
        plt.show()

    data_list = _get_datasets(choices=[3, 6])["choices"]
    d = get_dataset_for_eval_online(data_list, LIVER_PERCENT, TUMOR_PERCENT,
                                    True, (("hist", 200), ), False, config=Foo())
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

    # d = get_dataset_for_eval_online(data_list, LIVER_PERCENT, TUMOR_PERCENT,
    #                                 True, (("hist", 200), ), True, config=Foo())
    # # lst = None
    # # while True:
    # #     try:
    # #         f, l = next(gen)
    # #         cnt, lst = save(f, l, cnt, lst)
    # #     except StopIteration:
    # #         break
    # ff, ll = d.make_one_shot_iterator().get_next()
    # sess = tf.Session()
    # f, l = sess.run([ff, ll])
    # show(f, l, 0)

    # random.seed(1234)
    # np.random.seed(1234)
    # tf.set_random_seed(1234)
    # d = get_dataset_for_train(data_list, 0.66, 0.4, (0.8, 1.4),
    #                           False, (), True, 1., False, config=Foo())
    # ff, ll = d.make_one_shot_iterator().get_next()
    # sess = tf.Session()
    # cnt = 0
    # f, l = sess.run([ff, ll])
    # f, l = sess.run([ff, ll])
    # f, l = sess.run([ff, ll])
    # f, l = sess.run([ff, ll])
    # show(f, l, 3)
