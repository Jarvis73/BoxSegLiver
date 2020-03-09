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

RND_SEED = None  # For debug


# random.seed(1234)
# np.random.seed(1234)


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
    group.add_argument("--use_zscore", action="store_true", help="If use z-score, window/level will be disabled")
    group.add_argument("--use_gamma", action="store_true", help="Apply gamma transform for data augmentation")
    group.add_argument("--gamma_range", type=float, nargs="+", default=[0.7, 1.5])
    group.add_argument("--eval_in_patches", action="store_true")
    group.add_argument("--eval_num_batches_per_epoch", type=int, default=100)
    group.add_argument("--eval_mirror", action="store_true")

    group = parser.add_argument_group(title="G-Net Arguments")
    group.add_argument("--side_dropout", type=float, default=0.5, help="Dropout used in G-Net sub-networks")
    group.add_argument("--use_context", action="store_true")

    group.add_argument("--use_spatial", action="store_true")
    group.add_argument("--spatial_random", type=float, default=1.,
                       help="Probability of adding spatial guide to current slice with tumors "
                            "when use_spatial is on")
    group.add_argument("--eval_no_sp", action="store_true", help="No spatial guide in evaluation")
    group.add_argument("--min_std", type=float, default=2.,
                       help="Minimum stddev for spatial guide")
    group.add_argument("--save_sp_guide", action="store_true", help="Save spatial guide")
    group.add_argument("--use_se", action="store_true", help="Use SE-Block in G-Nets context guide module")
    group.add_argument("--eval_discount", type=float, default=0.85)
    group.add_argument("--eval_no_p", action="store_true", help="Evaluate with no propagation")
    group.add_argument("--real_sp", type=str, help="Path to real spatial guide.")
    group.add_argument("--guide_scale", type=float, default=5., help="Base scale of guides")
    group.add_argument("--ct_conv", default=1)
    group.add_argument("--case_id", type=int)
    group.add_argument("--pos", type=int, nargs="+")
    group.add_argument("--ct_base", type=int, default=[32], nargs="+", help="Context base size")


class Case(object):
    def __init__(self, item):
        if "vol_case" in item:
            self.imh, self.img = nii_kits.read_nii(item["vol_case"])
        else:
            print("Warning!!! no vol_case")
            self.imh, self.img = None, None

        if "lab_case" in item:
            self.lah, self.lab = nii_kits.read_nii(item["lab_case"], out_dtype=np.uint8)
        else:
            print("Warning!!! no lab_case")
            self.iah, self.lab = None, None


class Datasets(object):
    """ Load online """
    pickled = None
    data = {}

    def __getitem__(self, item):
        pid = item["PID"] if isinstance(item, dict) else item
        if pid not in self.data:
            self.data[pid] = Case(item)
        return self.data[pid]


def _get_datasets(test_fold=-1, mode="train", filter_size=10, choices=None, exclude=None, config=None):
    prepare_dir = Path(__file__).parent / "prepare"
    ds = Datasets()

    if ds.pickled is None:
        import pickle
        # Check existence
        obj_file = prepare_dir / "nf_analy.pkl"
        with obj_file.open("rb") as f:
            ds.pickled = pickle.load(f)

    meta = {x["PID"]: x for x in ds.pickled}

    def parse(case):
        t_slice = []
        for i, sli in enumerate(case[2]):
            small_t = [j for j in sli if sli[j]["area"] <= filter_size]
            for j in small_t:
                sli.pop(j)
            if len(sli) > 0:
                t_slice.append(i)
        case["t_slice"] = t_slice
        return case

    if not choices:
        # Load k_folds
        fold_path = prepare_dir / "k_folds.txt"
        all_cases = list(meta)
        if exclude:
            for exc in exclude:
                all_cases.remove(exc)
        print("Read:: 5 folds, test fold = %d" % test_fold)
        k_folds = misc.read_or_create_k_folds(fold_path, all_cases, k_split=5, seed=1357, verbose=False)
        if test_fold + 1 > len(k_folds):
            raise ValueError("test_fold too large")
        if test_fold < 0:
            raise ValueError("test_fold must be non-negative")
        testset = k_folds[test_fold]
        trainset = []
        for i, folds in enumerate(k_folds):
            if i != test_fold:
                trainset.extend(folds)

        if mode.lower() == "train":
            dataset_dict = []
            for idx in sorted([int(x) for x in trainset]):
                case = case = parse(meta[idx])
                if len(case["t_slice"]) > 0:
                    dataset_dict.append(case)
        elif mode.lower() == "eval" or mode.lower() == "eval_online":
            dataset_dict = []
            for idx in sorted([int(x) for x in testset]):
                case = parse(meta[idx])
                if len(case["t_slice"]) > 0 and case["PID"] != 70 and case["PID"] != 39:
                    dataset_dict.append(case)
        else:   # infer
            dataset_dict = []
            assert config
            assert config.case_id in meta, (config.case_id, list(meta.keys()))
            case = parse(meta[config.case_id])
            dataset_dict.append(case)
    else:
        dataset_dict = []
        for idx in choices:
            dataset_dict.append(parse(meta[idx]))

    return dataset_dict



def input_fn(mode, params):
    if "args" not in params:
        raise KeyError("params of input_fn need an \"args\" key")
    config = params["args"]

    dataset = _get_datasets(config.test_fold, mode, filter_size=config.filter_size, config=config)
    if len(dataset) == 0:
        raise ValueError("No valid dataset found!")
    logging.info("{}: {} NF MRIs ({} slices, {} tumors)"
                 .format(mode[:1].upper() + mode[1:], len(dataset),
                         sum([x["size"][0] for x in dataset]),
                         sum([len(x[3]) for x in dataset])))

    kwargs = {"mode": "Train"}
    with tf.variable_scope("InputPipeline"):
        if mode == ModeKeys.TRAIN:
            return get_dataset_for_train(dataset,
                                         tumor_percent=TUMOR_PERCENT,
                                         use_context=config.use_context,
                                         use_spatial=config.use_spatial,
                                         spatial_random=config.spatial_random,
                                         config=config,
                                         **kwargs)
        elif mode == "eval_online":
            kwargs["mode"] = "Eval_online"
            return get_dataset_for_eval_online(dataset,
                                               tumor_percent=0.,
                                               use_context=config.use_context,
                                               use_spatial=config.use_spatial,
                                               spatial_random=1.,
                                               config=config,
                                               **kwargs)
        elif mode == ModeKeys.EVAL:
            kwargs["mode"] = "Eval"
            return get_dataset_for_eval(dataset,
                                        use_context=config.use_context,
                                        use_spatial=config.use_spatial,
                                        config=config,
                                        **kwargs)
        elif mode == ModeKeys.PREDICT:
            kwargs["mode"] = "Infer"
            return get_image_patch(dataset,
                                   se_context=config.use_context,
                                   use_spatial=config.use_spatial,
                                   config=config,
                                   **kwargs)


#####################################
#
#   G-Net input pipeline
#
#####################################

def data_processing_cuda(features, labels, img_clip, config, mode):
    img0 = features["images"]            # [None, None, 3]
    if config.use_context:
        ct_coord = features["ct_coord"]     # [3, 4]
    sp_coord = features["sp_coord"]     # [4]

    if config.use_zscore:
        img0 = image_ops.zscore(img0)
    else:
        clip_min, clip_max = tf.split(img_clip, 2)
        img0 = (tf.clip_by_value(img0, clip_min, clip_max) - clip_min) / (clip_max - clip_min)

    img = tf.expand_dims(img0, axis=0)
    img = tf.image.resize_bilinear(img, (config.im_height, config.im_width), align_corners=True)
    img = tf.squeeze(img, axis=0)
    lab = tf.expand_dims(tf.expand_dims(labels, axis=0), axis=-1)
    lab = tf.image.resize_nearest_neighbor(lab, (config.im_height, config.im_width), align_corners=True)
    labels = tf.squeeze(lab, axis=[0, -1])

    c = config.im_channel // 2
    def ct_true():
        context = tf.image.crop_and_resize(tf.expand_dims(img0, axis=0)[..., c:c + 1], ct_coord, [0] * 3, [32, 32])
        context = tf.transpose(tf.squeeze(context, axis=-1), (1, 2, 0))
        return context

    if config.use_context:
        context = tf.cond(tf.reduce_max(ct_coord) > 0, ct_true,
                          lambda: tf.constant(0, dtype=tf.float32, shape=(32, 32, 3)))
        features["context"] = context

    def sp_true():
        center, stddev = tf.split(sp_coord, 2, axis=1)
        gd = image_ops.create_spatial_guide_2d(tf.shape(img)[:-1], center, stddev)
        return gd

    if config.use_spatial:
        spatial = tf.cond(tf.reduce_max(sp_coord) > 0, sp_true,
                          lambda: tf.zeros(tf.concat([tf.shape(img)[:-1], [1]], axis=0), tf.float32))
        features["sp_guide"] = spatial

    if config.use_gamma:
        img = image_ops.augment_gamma(img, gamma_range=config.gamma_range, retain_stats=True, p_per_sample=0.5)

    if config.noise_scale:
        img = image_ops.random_noise(img, config.noise_scale, seed=RND_SEED, ntype="normal")
        img *= tf.cast(tf.greater(tf.strings.length(im_files), 0), tf.float32)

    rf1 = config.random_flip & 1 > 0
    rf2 = config.random_flip & 2 > 0
    if (rf1 or rf2) and config.use_spatial:
        img = tf.concat((img, features["sp_guide"]), axis=-1)
    if rf1:
        img, labels = image_ops.random_flip_left_right(img, labels, RND_SEED)
        if config.use_context:
            features["context"] = image_ops.random_flip_left_right(features["context"], RND_SEED)
    if rf2:
        img, labels = image_ops.random_flip_up_down(img, labels, RND_SEED)
        if config.use_context:
            features["context"] = image_ops.random_flip_up_down(features["context"], RND_SEED)
    if (rf1 or rf2) and config.use_spatial:
        img, features["sp_guide"] = tf.split(img, [config.im_channel, 1], axis=-1)

    features["images"] = img
    return features, labels


# def gen_train_batch(data_list,
#                     batch_size,
#                     tumor_percent=0.,
#                     use_context=False,
#                     use_spatial=False,
#                     spatial_random=0.,
#                     config=None,
#                     **kwargs):
#     """ All coordinates are ij index """
#     # Load context guide if needed
#     z_score = kwargs.get("z_score", False)
#     use_gamma = kwargs.get("use_gamma", False)
#     random_noise = kwargs.get("random_noise", False)
#     random_flip_left_right = kwargs.get("random_flip_left_right", False)
#     random_flip_up_down = kwargs.get("random_flip_up_down", False)
#     random_scale = kwargs.get("random_scale", (1., 1.))
#     mode = kwargs.get("mode", "Train")
#
#     if z_score:
#         logging.info("{}: Use z-score transformation".format(mode))
#     if use_gamma:
#         logging.info("{}: Use gamma augmentation".format(mode))
#     if random_noise:
#         logging.info("{}: Add random noise, scale = {}".format(mode, config.noise_scale))
#     if random_flip_left_right:
#         logging.info("{}: Add random flip left <-> right".format(mode))
#     if random_flip_up_down:
#         logging.info("{}: Add random flip up <-> down".format(mode))
#     if random_scale[1] > random_scale[0]:
#         logging.info("{}: Add random zoom, scale = ({}, {})".format(mode, *random_scale))
#
#     d = data_list
#     list_of_keys = np.arange(len(d))
#
#     tumor_list_of_keys = []
#     for i in list_of_keys:
#         if len(d[i]["t_slice"]) > 0:
#             tumor_list_of_keys.append(i)
#
#     target_size = np.asarray((config.im_height, config.im_width), dtype=np.float32)
#     force_tumor = math.ceil(batch_size * tumor_percent)
#     empty_mmts = np.zeros((0, 2), dtype=np.float32)
#     ds = Datasets()
#
#     while True:
#         ci1 = np.random.choice(tumor_list_of_keys, force_tumor, True)
#         ci2 = np.random.choice(list_of_keys, batch_size - force_tumor, True)  # case indices
#         ci = np.concatenate((ci1, ci2), axis=0)
#
#         tumor_counter = 0
#         for j, i in enumerate(ci):
#             case = d[i]
#             ch, cw = list((target_size * np.random.uniform(*random_scale, size=2)).astype(np.int32))
#             depth, height, width = case["size"]
#             pid = case["PID"]
#
#             # Get selected slice
#             if tumor_counter < force_tumor:
#                 selected_slice = np.random.choice(case["t_slice"])
#                 tumor_counter += 1
#             else:
#                 selected_slice = random.randint(0, depth - 1)
#
#             # choice crop center
#             if selected_slice in case["t_slice"]:
#                 tar = case[2][selected_slice]
#                 t_id = np.random.choice(list(tar.keys()))
#                 c_id = np.random.choice(range(tar[t_id]["pix"].shape[0]))
#                 y, x = tar[t_id]["pix"][c_id]
#             else:
#                 t_id = 0
#                 y, x = random.randint(0, height - 1), random.randint(0, width - 1)
#             ofx = min(width - cw, max(0, x - cw // 2))
#             ofy = min(height - ch, max(0, y - ch // 2))
#
#             # Get multi-channel input
#             selected_slices = [selected_slice]
#             pre_emp, post_emp = 0, 0
#             if config.im_channel > 1:
#                 left_half_channel = (config.im_channel - 1) // 2
#                 for k in range(1, left_half_channel + 1):
#                     previous_slice = selected_slice - k
#                     if 0 <= previous_slice < depth:
#                         selected_slices.insert(0, previous_slice)
#                     else:
#                         pre_emp += 1
#                 right_half_channel = config.im_channel - 1 - left_half_channel
#                 for k in range(1, right_half_channel + 1):
#                     following_slice = selected_slice + k
#                     if 0 <= following_slice < depth:
#                         selected_slices.append(following_slice)
#                     else:
#                         post_emp += 1
#             img = ds[case].img[selected_slices, ofy:ofy + ch, ofx:ofx + cw].astype(np.float32)
#             if pre_emp or post_emp:
#                 img = np.concatenate([np.zeros((pre_emp,) + img.shape[1:], dtype=img.dtype),
#                                       img,
#                                       np.zeros((post_emp,) + img.shape[1:], dtype=img.dtype)], axis=0)
#             else:
#                 img = img.copy()
#             img = img.transpose((1, 2, 0))
#             lab = ds[case].lab[selected_slice, ofy:ofy + ch, ofx:ofx + cw]
#             if t_id:
#                 lab = (lab == t_id).astype(np.int32)
#
#             if z_score:
#                 array_kits.zscore(img)
#             else:  # Random clip image value
#                 _ = np.clip(img, 0, np.random.randn() * 30 + 900 if True else 900, out=img)
#                 img /= 900
#
#             features = {"names": pid}
#             # **********************************************************************************************************
#             use_guide = random.random() < spatial_random if use_context or use_spatial else None
#             if use_context:
#                 if use_guide:
#                     patches = []
#                     for s in [32, 64, 128]:
#                         ofx_ = min(width - 32, max(0, x - s // 2))
#                         ofy_ = min(height - 32, max(0, y - s // 2))
#                         patch = ds[case].img[selected_slice, ofy_:ofy_ + s, ofx_:ofx_ + s].astype(np.float32)
#                         if s > 32:
#                             patches.append(cv2.resize(patch, (32, 32), interpolation=cv2.INTER_LINEAR))
#                         else:
#                             patches.append(patch)
#                     ct = np.stack(patches, axis=-1)
#                     ct = (ct - ct.mean()) / (ct.std() + 1e-8)
#                 else:
#                     ct = np.zeros((32, 32, 3), dtype=np.float32)
#                 features["context"] = ct
#
#             if use_spatial:
#                 if use_guide:
#                     centers = [(y - ofy, x - ofx)]
#                     stddevs = [(config.guide_scale, config.guide_scale)]
#                     sp = array_kits.create_gaussian_distribution_v2([ch, cw], centers, stddevs, keepdims=True)
#                 else:
#                     sp = np.zeros((ch, cw, 1), dtype=np.float32)
#                 img = np.concatenate((img, sp), axis=-1)
#
#             # **********************************************************************************************************
#             # Augmentation
#             if use_gamma:
#                 im = img[..., :-1] if use_spatial else img
#                 im = array_kits.augment_gamma(im, gamma_range=(0.7, 1.5), retain_stats=True, p_per_sample=0.3)
#                 if use_spatial:
#                     img[..., :-1] = im
#             if random_noise:
#                 mask = img > 0
#                 img[mask] += np.random.randn()[mask] * config.noise_scale
#             if random_flip_left_right and random.random() > 0.5:
#                 img = img[:, ::-1]
#                 lab = lab[:, ::-1]
#                 if use_context:
#                     features["context"] = features["context"][:, ::-1]
#             if random_flip_up_down and random.random() > 0.5:
#                 img = img[::-1]
#                 lab = lab[::-1]
#                 if use_context:
#                     features["context"] = features["context"][::-1]
#             features["images"] = cv2.resize(img, (config.im_width, config.im_height), interpolation=cv2.INTER_LINEAR)
#             if use_spatial:
#                 features["sp_guide"] = features["images"][..., -1:]
#                 features["images"] = features["images"][..., :-1]
#             lab = cv2.resize(lab, (config.im_width, config.im_height), interpolation=cv2.INTER_NEAREST)
#
#             yield features, lab

def gen_train_batch(data_list,
                    batch_size,
                    tumor_percent=0.,
                    use_context=False,
                    use_spatial=False,
                    spatial_random=0.,
                    config=None, **kwargs):
    """ All coordinates are ij index """
    mode = kwargs.get("mode", "Train")
    train = mode == "Train"
    random_window_level = True

    if config.use_zscore:
        logging.info("{}: Use z-score transformation".format(mode))
    if train and config.use_gamma:
        logging.info("{}: Use gamma augmentation, range = ({}, {})".format(mode, *config.gamma_range))
    if train and config.noise_scale:
        logging.info("{}: Add random noise, scale = {}".format(mode, config.noise_scale))
    if train and config.random_flip & 1 > 0:
        logging.info("{}: Add random flip left <-> right".format(mode))
    if train and config.random_flip & 2 > 0:
        logging.info("{}: Add random flip up <-> down".format(mode))
    if train and config.zoom_scale[1] > config.zoom_scale[0]:
        logging.info("{}: Add random zoom, scale = ({}, {})".format(mode, *config.zoom_scale))

    d = data_list
    list_of_keys = np.arange(len(d))

    tumor_list_of_keys = []
    for i in list_of_keys:
        if len(d[i]["t_slice"]) > 0:
            tumor_list_of_keys.append(i)

    target_size = np.asarray((config.im_height, config.im_width), dtype=np.float32)
    force_tumor = math.ceil(batch_size * tumor_percent)
    empty_mmts = np.zeros((0, 2), dtype=np.float32)
    ds = Datasets()

    while True:
        ci1 = np.random.choice(tumor_list_of_keys, force_tumor, True)
        ci2 = np.random.choice(list_of_keys, batch_size - force_tumor, True)  # case indices
        ci = np.concatenate((ci1, ci2), axis=0)

        tumor_counter = 0
        for j, i in enumerate(ci):
            case = d[i]
            ch, cw = list((target_size * np.random.uniform(*config.zoom_scale, size=2)).astype(np.int32))
            depth, height, width = case["size"]
            pid = case["PID"]

            # Get selected slice
            if tumor_counter < force_tumor:
                selected_slice = np.random.choice(case["t_slice"])
                tumor_counter += 1
            else:
                selected_slice = random.randint(0, depth - 1)

            # choice crop center
            if selected_slice in case["t_slice"]:
                tar = case[2][selected_slice]
                t_id = np.random.choice(list(tar.keys()))
                c_id = np.random.choice(range(tar[t_id]["pix"].shape[0]))
                y, x = tar[t_id]["pix"][c_id]
            else:
                t_id = 0
                y, x = random.randint(0, height - 1), random.randint(0, width - 1)
            ofx = min(width - cw, max(0, x - cw // 2))
            ofy = min(height - ch, max(0, y - ch // 2))

            # Get multi-channel input
            selected_slices = [selected_slice]
            pre_emp, post_emp = 0, 0
            if config.im_channel > 1:
                left_half_channel = (config.im_channel - 1) // 2
                for k in range(1, left_half_channel + 1):
                    previous_slice = selected_slice - k
                    if 0 <= previous_slice < depth:
                        selected_slices.insert(0, previous_slice)
                    else:
                        pre_emp += 1
                right_half_channel = config.im_channel - 1 - left_half_channel
                for k in range(1, right_half_channel + 1):
                    following_slice = selected_slice + k
                    if 0 <= following_slice < depth:
                        selected_slices.append(following_slice)
                    else:
                        post_emp += 1
            img = ds[case].img[selected_slices, ofy:ofy + ch, ofx:ofx + cw].astype(np.float32)
            if pre_emp or post_emp:
                img = np.concatenate([np.zeros((pre_emp,) + img.shape[1:], dtype=img.dtype),
                                      img,
                                      np.zeros((post_emp,) + img.shape[1:], dtype=img.dtype)], axis=0)
            img = img.transpose((1, 2, 0))
            if t_id:
                lab = (ds[case].lab[selected_slice, ofy:ofy + ch, ofx:ofx + cw] == t_id).astype(np.int32)
            else:
                lab = np.zeros((ch, cw), dtype=np.int32)

            if config.use_zscore:
                img_clip = (GRAY_MIN, GRAY_MAX)
            elif random_window_level:  # Random clip image value
                # img_clip = (0, random.randint(700, 1000))
                img_clip = (0, random.randint(500, 700))
            else:
                # img_clip = (0, 850)
                img_clip = (0, 600)

            features = {"names": pid, "images": img}
            # **********************************************************************************************************
            use_guide = random.random() < spatial_random if use_context or use_spatial else None
            if use_context:
                cb = config.ct_base
                ct_size = cb if len(cb) == 3 else [cb[0], cb[0] * 2, cb[0] * 4]
                if use_guide:
                    patches = []
                    for s in ct_size:
                        ofx_ = min(width - s, max(0, x - s // 2)) - ofx
                        ofy_ = min(height - s, max(0, y - s // 2)) - ofy
                        patches.append((ofy_ / (ch - 1), ofx_ / (cw - 1), (ofy_ + s) / (ch - 1), (ofx_ + s) / (cw - 1)))
                else:
                    patches = [(0, 0, 0, 0)] * 3
                features["ct_coord"] = patches

            if use_spatial:
                if use_guide:
                    sp_coord = [(y - ofy) / (ch - 1) * (config.im_height - 1),
                                (x - ofx) / (cw - 1) * (config.im_width - 1),
                                config.guide_scale, config.guide_scale]
                else:
                    sp_coord = [0, 0, 0, 0]
                features["sp_coord"] = [sp_coord]

            yield features, lab, img_clip


def get_dataset_for_train(data_list,
                          tumor_percent=0.,
                          use_context=False,
                          use_spatial=False,
                          spatial_random=0.,
                          config=None, **kwargs):
    batch_size = distribution_utils.per_device_batch_size(config.batch_size, config.num_gpus)

    def train_gen():
        return gen_train_batch(data_list, batch_size, tumor_percent,
                               use_context=use_context,
                               use_spatial=use_spatial,
                               spatial_random=spatial_random,
                               config=config,
                               **kwargs)

    output_types = ({"images": tf.float32, "names": tf.int32}, tf.int32, tf.float32)
    output_shapes = ({"images": tf.TensorShape([None, None, config.im_channel]), "names": tf.TensorShape([])},
                     tf.TensorShape([None, None]), tf.TensorShape([2]))
    if use_context:
        logging.info("Train: Use context guide")
        output_types[0]["ct_coord"] = tf.float32
        output_shapes[0]["ct_coord"] = tf.TensorShape([3, 4])
    if use_spatial:
        logging.info("Train: Use spatial guide")
        output_types[0]["sp_coord"] = tf.float32
        output_shapes[0]["sp_coord"] = tf.TensorShape([1, 4])

    dataset = (tf.data.Dataset.from_generator(train_gen, output_types, output_shapes)
               .apply(tf.data.experimental.map_and_batch(
                    lambda *args_: data_processing_cuda(*args_, config=config, mode="Train"),
                    batch_size=batch_size, num_parallel_batches=1))
               .prefetch(buffer_size=contrib_data.AUTOTUNE))

    return dataset


def get_dataset_for_eval_online(data_list,
                                tumor_percent=0.,
                                use_context=False,
                                use_spatial=False,
                                spatial_random=0.,
                                config=None, **kwargs):
    batch_size = distribution_utils.per_device_batch_size(config.batch_size, config.num_gpus)

    infinity_generator = gen_train_batch(data_list, batch_size, tumor_percent,
                                         use_context=use_context,
                                         use_spatial=use_spatial,
                                         spatial_random=spatial_random,
                                         config=config,
                                         **kwargs)

    def eval_2d_gen():
        for _ in tqdm.tqdm(range(config.eval_num_batches_per_epoch * config.batch_size)):
            yield next(infinity_generator)

    output_types = ({"images": tf.float32, "names": tf.int32}, tf.int32, tf.float32)
    output_shapes = ({"images": tf.TensorShape([None, None, config.im_channel]), "names": tf.TensorShape([])},
                     tf.TensorShape([None, None]), tf.TensorShape([2]))
    if use_context:
        logging.info("Eval_online: Use context guide")
        output_types[0]["ct_coord"] = tf.float32
        output_shapes[0]["ct_coord"] = tf.TensorShape([3, 4])
    if use_spatial:
        logging.info("Eval_online: Use spatial guide")
        output_types[0]["sp_coord"] = tf.float32
        output_shapes[0]["sp_coord"] = tf.TensorShape([1, 4])

    dataset = (tf.data.Dataset.from_generator(eval_2d_gen, output_types, output_shapes)
               .apply(tf.data.experimental.map_and_batch(
                    lambda *args_: data_processing_cuda(*args_, config=config, mode="Eval_online"),
                    batch_size=batch_size, num_parallel_batches=1))
               .prefetch(buffer_size=contrib_data.AUTOTUNE))

    return dataset


def get_dataset_for_eval(data_list,
                         use_context=False,
                         use_spatial=False,
                         config=None, **kwargs):
    """ Batch size = 1 """
    mode = kwargs.get("mode", "Eval")
    if config.use_zscore:
        logging.info("{}: Use z-score transformation".format(mode))

    d = data_list
    ds = Datasets()
    bs = config.batch_size
    target_size = np.asarray((config.im_height, config.im_width), dtype=np.float32)
    disc = ndi.generate_binary_structure(2, 2)

    for ci, case in enumerate(d[config.eval_skip_num:]):
        ch, cw = target_size
        depth, height, width = case["size"]
        pid = case["PID"]
        img3d = ds[case].img

        if config.filter_size > 0:
            lab3d = ds[case].lab
            for ti, t in case[3].items():
                if t["area"] < config.filter_size:
                    z, y, x = t["pix"].transpose()
                    lab3d[z, y, x] = 0
        else:
            lab3d = ds[case].lab
        lab3d = np.clip(lab3d, 0, 1)

        features = {"names": pid, "mirror": 0, "pad": 0}

        bi = 0
        yield None, case
        # For each tumor
        for si, s_info in enumerate(case[2]):
            # Get multi-channel input
            left_half_channel = (config.im_channel - 1) // 2
            right_half_channel = config.im_channel - 1 - left_half_channel
            pre_emp = max(0, si - left_half_channel) - (si - left_half_channel)
            post_emp = max(0, depth - 1 - si - right_half_channel) - (depth - 1 - si - right_half_channel)
            selected_slices = list(range(si - left_half_channel + pre_emp, si + 1 + right_half_channel - post_emp))

            for ti, t in s_info.items():
                centers = []
                y1, x1, y2, x2 = t["bb"]
                # compute median of this tumor in this slice
                median = np.median(t["pix"], axis=0).astype(np.int32)
                if not np.isclose(np.sum((t["pix"] - median) ** 2, axis=1), 0, atol=2.1).any():
                    # centroid not in pixel list
                    labeled_patch, n = ndi.label(img3d[si, y1:y2, x1:x2], disc)
                    if n == 1:
                        # Wired shape, we randomly choice seed
                        c_id = np.random.choice(range(t["pix"].shape[0]))
                        y, x = t["pix"][c_id]
                        centers.append((y, x))
                    else:
                        # Compute centroid for each part of this object
                        for i in range(1, n):
                            y, x = np.median(np.array(np.where(labeled_patch == i)), axis=1).astype(np.int32)
                            y, x = y + y1, x + x1
                            centers.append((y, x))
                else:
                    centers.append(tuple(median))
                # labeled_patch, n = ndi.label(img3d[si, y1:y2, x1:x2], disc)
                # if n == 1:
                #     # Wired shape, we randomly choice seed
                #     c_id = np.random.choice(range(t["pix"].shape[0]))
                #     y, x = t["pix"][c_id]
                #     centers.append((y, x))
                # else:
                #     # Compute centroid for each part of this object
                #     for i in range(1, n):
                #         y, x = np.median(np.array(np.where(labeled_patch == i)), axis=1).astype(np.int32)
                #         y, x = y + y1, x + x1
                #         centers.append((y, x))

                    # compute bounding box
                expand = 5
                y1 = max(y1 - expand, 0)
                x1 = max(x1 - expand, 0)
                y2 = min(y2 + expand, height)
                x2 = min(x2 + expand, width)
                cy, cx = y2 - y1, x2 - x1   # crop_y, crop_x
                if cy < config.im_height:
                    half_h = config.im_height // 2
                    y_center = (y1 + y2 - 1) // 2
                    y_center = min(height - half_h, max(half_h, y_center))
                    y1, y2 = y_center - half_h, y_center + half_h
                    cy = y2 - y1
                if cx < config.im_width:
                    half_w = config.im_width // 2
                    x_center = (x1 + x2 - 1) // 2
                    x_center = min(width - half_w, max(half_w, x_center))
                    x1, x2 = x_center - half_w, x_center + half_w
                    cx = x2 - x1
                centers = np.array(centers, dtype=np.float32) - [y1, x1]
                stddevs = np.ones_like(centers, dtype=np.float32) * config.guide_scale
                img = img3d[selected_slices, y1:y2, x1:x2].astype(np.float32)

                if pre_emp or post_emp:
                    img = np.concatenate([np.zeros((pre_emp,) + img.shape[1:], dtype=img.dtype),
                                          img,
                                          np.zeros((post_emp,) + img.shape[1:], dtype=img.dtype)], axis=0)
                img = img.transpose((1, 2, 0))

                if config.use_zscore:
                    img = (img - img.mean()) / (img.std() + 1e-8)
                else:
                    # img = np.clip(img, 0, 850) / 850
                    img = np.clip(img, 0, 600) / 600

                # spatial/context guides
                if config.use_spatial:
                    sp = array_kits.create_gaussian_distribution_v2([cy, cx], centers, stddevs)
                    sp = cv2.resize(sp, (config.im_width, config.im_height), interpolation=cv2.INTER_LINEAR)
                    if bi == 0:
                        features["sp_guide"] = np.zeros((bs, config.im_height, config.im_width, 1), dtype=np.float32)
                    features["sp_guide"][bi, :, :, 0] = sp

                if config.use_context:
                    patches = []
                    # We choose the middle centroid
                    mid = np.median(centers, axis=0)
                    idx = np.argmin(np.sum((centers - mid) ** 2, axis=1))
                    yy, xx = centers[idx].astype(np.int32)
                    cb = config.ct_base
                    ct_size = cb if len(cb) == 3 else [cb[0], cb[0] * 2, cb[0] * 4]
                    for s in ct_size:
                        ofx_ = min(cx - s, max(0, xx - s // 2))
                        ofy_ = min(cy - s, max(0, yy - s // 2))
                        patch = img[ofy_:ofy_ + s, ofx_:ofx_ + s, config.im_channel // 2].astype(np.float32)
                        if s > 32:
                            patches.append(cv2.resize(patch, (32, 32), interpolation=cv2.INTER_LINEAR))
                        else:
                            patches.append(patch)
                    ct = np.stack(patches, axis=-1)
                    ct = (ct - ct.mean()) / (ct.std() + 1e-8)
                    if bi == 0:
                        features["context"] = np.zeros((bs, 32, 32, 3), dtype=np.float32)
                    features["context"][bi] = ct

                if cx != config.im_width or cy != config.im_height:
                    img = cv2.resize(img, (config.im_width, config.im_height), interpolation=cv2.INTER_LINEAR)

                if bi == 0:
                    features["images"] = np.zeros((bs, config.im_height, config.im_width, 3), dtype=np.float32)
                    features["sid"] = np.ones((bs,), dtype=np.int32) * -1
                    features["bb"] = np.ones((bs, 4), dtype=np.int32) * -1
                features["images"][bi] = img
                features["sid"][bi] = si
                features["bb"][bi] = [y1, x1, y2, x2]

                if bi == bs - 1:
                    for feat, seg in eval_batch_generator(features, config, lab3d):
                        yield feat, seg
                    bi = 0
                else:
                    bi += 1

        if bi > 0:  # Final batch (if have)
            features["pad"] = bs - bi
            for feat, seg in eval_batch_generator(features, config, lab3d):
                yield feat, seg

        yield None, lab3d


def get_image_patch(data_list,
                    use_context=False,
                    use_spatial=False,
                    config=None, **kwargs):
    """ Batch size = 1 """
    mode = kwargs.get("mode", "Infer")
    if config.use_zscore:
        logging.info("{}: Use z-score transformation".format(mode))

    ds = Datasets()
    bs = config.batch_size
    target_size = np.asarray((config.im_height, config.im_width), dtype=np.float32)
    disc = ndi.generate_binary_structure(2, 2)

    case = data_list[0]
    ch, cw = target_size
    depth, height, width = case["size"]
    pid = case["PID"]
    img3d = ds[case].img
    features = {"names": pid}

    si, s_info = config.pos[0], case[2][config.pos[0]]
    # Get multi-channel input
    left_half_channel = (config.im_channel - 1) // 2
    right_half_channel = config.im_channel - 1 - left_half_channel
    pre_emp = max(0, si - left_half_channel) - (si - left_half_channel)
    post_emp = max(0, depth - 1 - si - right_half_channel) - (depth - 1 - si - right_half_channel)
    selected_slices = list(range(si - left_half_channel + pre_emp, si + 1 + right_half_channel - post_emp))

    y, x = config.pos[1:]
    # compute bounding box
    expand = config.im_height // 2
    y1 = max(y - expand, 0)
    x1 = max(x - expand, 0)
    y2 = min(y + expand, height)
    x2 = min(x + expand, width)
    cy, cx = y2 - y1, x2 - x1   # crop_y, crop_x
    if cy < config.im_height:
        half_h = config.im_height // 2
        y_center = (y1 + y2 - 1) // 2
        y_center = min(height - half_h, max(half_h, y_center))
        y1, y2 = y_center - half_h, y_center + half_h
        cy = y2 - y1
    if cx < config.im_width:
        half_w = config.im_width // 2
        x_center = (x1 + x2 - 1) // 2
        x_center = min(width - half_w, max(half_w, x_center))
        x1, x2 = x_center - half_w, x_center + half_w
        cx = x2 - x1
    centers = np.array([[y, x]], dtype=np.float32) - [y1, x1]
    stddevs = np.ones_like(centers, dtype=np.float32) * config.guide_scale
    img = img3d[selected_slices, y1:y2, x1:x2].astype(np.float32)

    if pre_emp or post_emp:
        img = np.concatenate([np.zeros((pre_emp,) + img.shape[1:], dtype=img.dtype),
                              img,
                              np.zeros((post_emp,) + img.shape[1:], dtype=img.dtype)], axis=0)
    img = img.transpose((1, 2, 0))

    if config.use_zscore:
        img = (img - img.mean()) / (img.std() + 1e-8)
    else:
        img = np.clip(img, 0, 850) / 850

    # spatial/context guides
    if config.use_spatial:
        sp = array_kits.create_gaussian_distribution_v2([cy, cx], centers, stddevs)
        sp = cv2.resize(sp, (config.im_width, config.im_height), interpolation=cv2.INTER_LINEAR)
        features["sp_guide"] = sp[None, :, :, None]

    if config.use_context:
        patches = []
        # We choose the middle centroid
        mid = np.median(centers, axis=0)
        idx = np.argmin(np.sum((centers - mid) ** 2, axis=1))
        yy, xx = centers[idx].astype(np.int32)
        cb = config.ct_base
        ct_size = cb if len(cb) == 3 else [cb[0], cb[0] * 2, cb[0] * 4]
        for s in ct_size:
            ofx_ = min(cx - s, max(0, xx - s // 2))
            ofy_ = min(cy - s, max(0, yy - s // 2))
            patch = img[ofy_:ofy_ + s, ofx_:ofx_ + s, config.im_channel // 2].astype(np.float32)
            if s > 32:
                patches.append(cv2.resize(patch, (32, 32), interpolation=cv2.INTER_LINEAR))
            else:
                patches.append(patch)
        ct = np.stack(patches, axis=-1)
        ct = (ct - ct.mean()) / (ct.std() + 1e-8)
        features["context"] = ct[None]

    if cx != config.im_width or cy != config.im_height:
        img = cv2.resize(img, (config.im_width, config.im_height), interpolation=cv2.INTER_LINEAR)

    features["images"] = img[None]
    features["bb"] = [[y1, x1, y2, x2]]

    return features, None


def eval_batch_generator(features, config, labels=None):
    yield copy.copy(features), labels

    if config.eval_mirror:
        if config.random_flip & 1 > 0:
            tmp = copy.copy(features)
            tmp["images"] = np.flip(tmp["images"], axis=2)
            if config.use_spatial:
                tmp["sp_guide"] = np.flip(tmp["sp_guide"], axis=2)
            tmp["mirror"] = 1
            yield tmp, labels
        if config.random_flip & 2 > 0:
            tmp = copy.copy(features)
            tmp["images"] = np.flip(tmp["images"], axis=1)
            if config.use_spatial:
                tmp["sp_guide"] = np.flip(tmp["sp_guide"], axis=1)
            tmp["mirror"] = 2
            yield tmp, labels
        if config.random_flip & 3 > 0:
            tmp = copy.copy(features)
            tmp["images"] = np.flip(np.flip(tmp["images"], axis=2), axis=1)
            if config.use_spatial:
                tmp["sp_guide"] = np.flip(np.flip(tmp["sp_guide"], axis=2), axis=1)
            tmp["mirror"] = 3
            yield tmp, labels


if __name__ == "__main__":
    import matplotlib

    matplotlib.use("Qt5Agg")
    import matplotlib.pyplot as plt

    random.seed(1234)
    np.random.seed(1234)
    tf.set_random_seed(1234)


    class Foo(object):
        batch_size = 8
        im_height = 256
        im_width = 256
        im_channel = 3
        num_gpus = 1
        noise_scale = 0.
        random_flip = 3
        min_std = 2.
        use_zscore = True
        use_gamma = True
        zoom_scale = (1.0, 1.25)
        guide_scale = 5.
        use_context = True
        use_spatial = True
        eval_skip_num = 0
        eval_mirror = False
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
        if "context" in f:
            plt.subplot(236)
            max_val = f["context"].max()
            plt.imshow(np.hstack((f["context"][idx, :, :, 0],
                                  np.ones((32, 2)) * max_val,
                                  f["context"][idx, :, :, 1],
                                  np.ones((32, 2)) * max_val,
                                  f["context"][idx, :, :, 2],)), cmap="gray")
        plt.show()


    # data_list = _get_datasets(choices=[36])["choices"]
    data_list = _get_datasets(0, mode="val", filter_size=0)

    # d = get_dataset_for_train(data_list, TUMOR_PERCENT, True, True, 1.0, config=Foo())
    # ff, ll = d.make_one_shot_iterator().get_next()
    # sess = tf.Session()
    # f, l = sess.run([ff, ll])

    # d = get_dataset_for_eval(data_list, True, True, config=Foo(), mode="Eval")
    # f, l = next(d)
    #
    # print(f["context"].max(), f["context"].min())
    # show(f, l, 0)
