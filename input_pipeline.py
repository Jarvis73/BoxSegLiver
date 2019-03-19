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

import numpy as np
import tensorflow as tf
from functools import partial
from pathlib import Path
from tensorflow.python.platform import tf_logging as logging

from utils import image_ops
from utils import array_kits

Dataset = tf.data.Dataset
ModeKeys = tf.estimator.ModeKeys

# number for debug, None for training
SEED_FILE = 3456
SEED_BATCH = 7890
SEED_WIDTH = 3489
SEED_LEVEL = 8062

# record block length
BLOCK_LENGTH = 1

# shuffle buffer size
SHUFFLE_BUFFER_SIZE = 160

# Preprocess name scope
PREPROCESS = "Preprocess/"


def add_arguments(parser):
    group = parser.add_argument_group(title="Input Pipeline Arguments")
    group.add_argument("--dataset_for_train",
                       type=str,
                       nargs="*",
                       required=False, help="TFRecord names for model input. Such as "
                                            "\"LiTS/records/data_fold_0.tfrecord\" or a *.json file "
                                            "(Check data/dataset_example.json for details)")
    group.add_argument("--dataset_for_eval_while_train",
                       type=str,
                       nargs="*",
                       required=False, help="Same as --dataset_for_train")
    group.add_argument("--dataset_for_eval",
                       type=str,
                       nargs="*",
                       required=False, help="Same as --dataset_for_train")
    group.add_argument("--dataset_for_train_boxes",
                       type=str,
                       nargs="*",
                       required=False, help="Same as --dataset_for_train")
    group.add_argument("--dataset_for_eval_while_train_boxes",
                       type=str,
                       nargs="*",
                       required=False, help="Same as --dataset_for_train")
    group.add_argument("--im_height",
                       type=int,
                       default=512,
                       required=False, help="Image height (default: %(default)d)")
    group.add_argument("--im_width",
                       type=int,
                       default=512,
                       required=False, help="Image width (default: %(default)d)")
    group.add_argument("--im_channel",
                       type=int,
                       default=1,
                       required=False, help="Image channel (default: %(default)d)")
    group.add_argument("--triplet",     # TODO(ZJW): Deprecated, use --input_group 3 instead
                       action="store_true",
                       required=False, help="Use triplet as inputs")
    group.add_argument("--input_group",
                       type=int,
                       default=1,
                       choices=[1, 3, 5],
                       required=False, help="Use single slice or triplet or quintuplet... as inputs")
    group.add_argument("--resize_for_batch",
                       action="store_true",
                       required=False, help="Resize image to the same shape for batching")
    group.add_argument("--w_width",
                       type=float,
                       default=450,
                       required=False, help="Medical image window width (default: %(default)f)")
    group.add_argument("--w_level",
                       type=float,
                       default=25,
                       required=False, help="Medical image window level (default: %(default)f)")
    group.add_argument("--flip",
                       action="store_true",
                       required=False, help="Augment dataset with random flip")
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
    group.add_argument("--use_spatial_guide",
                       action="store_true",
                       required=False, help="Use spatial guide")
    group.add_argument("--random_spatial_guide",
                       action="store_true",
                       required=False, help="Randomly show spatial guide or not")
    group.add_argument("--use_fake_guide",
                       action="store_true",
                       required=False, help="Use fake spatial guide for better generalization")
    group.add_argument("--fake_rate",
                       type=float,
                       default=1.0,
                       required=False, help="#Fake / #Real (default: %(default)f)")
    group.add_argument("--center_perturb",
                       type=float,
                       default=0.2,
                       required=False, help="Center perturbation scale for spatial guide (default: %(default)f)")
    group.add_argument("--stddev_perturb",
                       type=float,
                       default=0.4,
                       required=False, help="stddev perturbation scale for spatial guide (default: %(default)f)")
    group.add_argument("--only_tumor",
                       action="store_true",
                       required=False, help="Training tumor segmentation and compute loss with liver mask")
    group.add_argument("--filter_size",
                       type=int,
                       default=100,
                       required=False, help="Input pipeline example filter for removing small size objects "
                                            "(default: %(default)d)")
    group.add_argument("--case_weights",
                       action="store_true",
                       required=False, help="Add cases' weights to loss for balancing dataset. If set, "
                                            "'extra/weights' will be parsed from example proto")


def _collect_datasets(datasets):
    tf_records = []

    if not isinstance(datasets, (list, tuple)):
        raise TypeError("`datasets` must be iterable, got {}".format(type(datasets)))

    for x in datasets:
        record = Path(__file__).parent / "data" / x  # Check exists in config.check_args()
        if record.suffix == ".tfrecord":
            tf_records.append(str(record))
        elif record.suffix == ".json":
            import json
            with record.open() as f:
                records = json.load(f)["dataset"]
            for one_file in records:
                record_ = Path(__file__).parent / "data" / one_file
                if record_.suffix == ".tfrecord":
                    tf_records.append(str(record_))
                else:
                    raise ValueError("Not supported data format: " + str(record_))
        else:
            raise ValueError("Not supported data format: " + str(record))

    return tf_records


def input_fn(mode, params):
    if "args" not in params:
        raise KeyError("params of input_fn need an \"args\" key")

    args = params["args"]
    tf_records = _collect_datasets(eval("args.dataset_for_" + mode))
    if len(tf_records) == 0:
        raise ValueError("No valid dataset found!")

    with tf.variable_scope("InputPipeline"):
        if mode == ModeKeys.TRAIN:
            logging.info("Train: {}".format(tf_records))
            if args.cls_branch:
                box_records = _collect_datasets(args.dataset_for_train_boxes)
                logging.info("Train: {}".format(box_records))
                return get_2d_images_and_bboxes_dataset_for_train(tf_records, box_records, args)
            return get_2d_multi_records_dataset_for_train(tf_records, args)
        elif mode == "eval_while_train":
            logging.info("Eval WT: {}".format(tf_records))
            if args.eval_3d:
                return get_3d_multi_records_dataset_for_eval(tf_records, mode, args)
            else:
                if args.cls_branch:
                    box_records = _collect_datasets(args.dataset_for_eval_while_train_boxes)
                    logging.info("Eval: {}".format(box_records))
                    return get_2d_images_and_bboxes_dataset_for_eval(tf_records, box_records, args)
                return get_2d_multi_records_dataset_for_eval(tf_records, args)
        elif mode == ModeKeys.EVAL:
            # For 3D
            logging.info("Eval: {}".format(tf_records))
            if args.cls_branch:
                return get_3d_images_and_bboxes_dataset_for_eval(tf_records, args)
            return get_3d_multi_records_dataset_for_eval(tf_records, args)


def filter_slices(*example_proto, args=None, strategy="empty", **kwargs):
    """
    Filter slices with specified strategy

    Parameters
    ----------
    example_proto: proto buffer
        TF-Example
    args: ArgumentParser
        Used arguments: w_width, w_level
    strategy: str
        Support `empty` and `area`
    kwargs: dict
        If strategy is `area`, then keyword argument `size` is necessary as a threshold.

    Returns
    -------
    True for keeping and False for discarding
    """
    if strategy not in ["empty", "area"]:
        raise ValueError("Not supported filter strategy: {}".format(strategy))
    if strategy == "area":
        if "size" not in kwargs or not isinstance(kwargs["size"], int):
            raise KeyError("If strategy is `area`, then keyword argument `size` with type "
                           "int is needed.")
        if kwargs["size"] < 0:
            raise ValueError("Argument size can not be negative, got {}".format(kwargs["size"]))

    with tf.name_scope("DecodeProto/"):
        if strategy == "empty":
            features = {"extra/empty": tf.FixedLenFeature([], tf.int64)}
            if args.case_weights:
                features["extra/weights"] = tf.FixedLenFeature([], tf.float32)
            features = tf.parse_single_example(example_proto[0], features=features)

            cond1 = tf.equal(features["extra/empty"], 0)
            if args.case_weights:
                cond2 = tf.less(tf.random.uniform((), dtype=tf.float32), features["extra/weights"])
                return tf.logical_and(cond1, cond2)
        else:   # strategy == "area"
            features = {"segmentation/encoded": tf.FixedLenFeature([], tf.string)}
            if args.case_weights:
                features["extra/weights"] = tf.FixedLenFeature([], tf.float32)
            features = tf.parse_single_example(example_proto[0], features=features)

            label = tf.decode_raw(features["segmentation/encoded"], tf.uint8, name="DecodeMask")
            if args.only_tumor:
                obj_label = tf.clip_by_value(tf.to_int32(label) - 1, 0, 1)
            else:
                obj_label = label
            num_dense = tf.count_nonzero(obj_label)

            cond1 = tf.greater_equal(num_dense, kwargs["size"])
            if args.case_weights:
                cond2 = tf.less(tf.random.uniform((), dtype=tf.float32), features["extra/weights"])
                return tf.logical_and(cond1, cond2)

        return cond1


def parse_2d_example_proto(example_proto, mode, args):
    """
    Parse example proto

    Parameters
    ----------
    example_proto: protobuf
        TF-Example
    mode: str
    args: ArgumentParser
        Used arguments: w_width, w_level

    Returns
    -------
    features and labels

    """
    logging.info("Parse 2D example")
    with tf.name_scope("DecodeProto/"):
        features = {
            "image/encoded": tf.FixedLenFeature([], tf.string),
            "image/shape": tf.FixedLenFeature([3], tf.int64),
            "image/name": tf.FixedLenFeature([], tf.string),
            "segmentation/encoded": tf.FixedLenFeature([], tf.string),
            "segmentation/shape": tf.FixedLenFeature([3 if args.only_tumor else 2], tf.int64),
            "extra/number": tf.FixedLenFeature([], tf.int64),
        }
        features = tf.parse_single_example(example_proto, features=features)

        with tf.name_scope(PREPROCESS):
            image = tf.decode_raw(features["image/encoded"], tf.int16, name="DecodeImage")
            image = tf.reshape(image, features["image/shape"], name="ReshapeImage")
            image = tf.to_float(image)
            label = tf.decode_raw(features["segmentation/encoded"], tf.uint8, name="DecodeMask")
            label = tf.reshape(label, features["segmentation/shape"], name="ReshapeMask")
            label = tf.to_int32(label)
            if len(args.classes) == 1 and not args.only_tumor:  # Only liver
                label = tf.clip_by_value(label, 0, 1)

            if mode == "train":
                image = image_ops.random_adjust_window_width_level(image, args.w_width, args.w_level)
            else:
                image = image_ops.adjust_window_width_level(image, args.w_width, args.w_level)

            ret_features = {"images": image,
                            "names": features["image/name"]}

            if hasattr(args, "train_without_eval") and not args.train_without_eval:
                ret_features["pads"] = tf.constant(0, dtype=tf.int64)
            if args.resize_for_batch:
                ret_features["bboxes"] = tf.constant([0] * 6, dtype=tf.int64)
            if args.cls_branch:
                ret_features["im_info"] = (features["segmentation/shape"][:-1]
                                           if args.only_tumor else features["segmentation/shape"])
            if args.case_weights:
                ret_features["weights"] = features["extra/weights"]

    # return features and labels
    return ret_features, label


def parse_3d_example_proto(example_proto, args):
    """
    Parse example proto

    Parameters
    ----------
    example_proto
    args: ArgumentParser
        Used arguments: w_width, w_level

    Returns
    -------
    features and labels

    """
    logging.info("Parse 3D example")
    with tf.name_scope("DecodeProto/"):
        features = {
            "image/encoded": tf.FixedLenFeature([], tf.string),
            "image/shape": tf.FixedLenFeature([4], tf.int64),
            "image/name": tf.FixedLenFeature([], tf.string),
            "segmentation/encoded": tf.FixedLenFeature([], tf.string),
            "segmentation/shape": tf.FixedLenFeature([3], tf.int64),
        }
        if args.resize_for_batch:
            features["extra/bbox"] = tf.FixedLenFeature([6], tf.int64)
        features = tf.parse_single_example(example_proto, features=features)

        image = tf.decode_raw(features["image/encoded"], tf.int16, name="DecodeImage")
        image = tf.reshape(image, features["image/shape"], name="ReshapeImage")
        image = tf.to_float(image)
        label = tf.decode_raw(features["segmentation/encoded"], tf.uint8, name="DecodeMask")
        label = tf.reshape(label, features["segmentation/shape"], name="ReshapeMask")
        label = tf.to_int32(label)
        if len(args.classes) == 1 and not args.only_tumor:  # Only liver
            label = tf.clip_by_value(label, 0, 1)

    with tf.name_scope(PREPROCESS):
        # image shape [depth, height, width, channel]
        image = image_ops.adjust_window_width_level(image, args.w_width, args.w_level)
        # Padding volume with empty slices to fit batch_size
        bs = args.batch_size
        pad = (bs - (features["image/shape"][0] % bs)) % bs
        shape_img = tf.concat(([pad], features["image/shape"][1:]), axis=0)
        shape_lab = tf.concat(([pad], features["segmentation/shape"][1:]), axis=0)
        zero_float = tf.zeros(shape_img, dtype=image.dtype)
        zero_int64 = tf.zeros(shape_lab, dtype=label.dtype)
        image = tf.concat((image, zero_float), axis=0)
        label = tf.concat((label, zero_int64), axis=0)

        if args.only_tumor:
            # Remove background
            image = image * tf.expand_dims(tf.cast(tf.clip_by_value(label, 0, 1), image.dtype),
                                           axis=-1)

        ret_features = {"images": image,
                        "names": features["image/name"],
                        "pads": pad}

        if args.resize_for_batch:
            ret_features["bboxes"] = features["extra/bbox"]
        if args.cls_branch:
            ret_features["im_info"] = features["segmentation/shape"][1:]

    return ret_features, label


def parse_bb_example_proto(example_proto, args):
    """
    Parse example proto

    Parameters
    ----------
    example_proto
    args: ArgumentParser
        Used arguments: w_width, w_level

    Returns
    -------
    features and labels

    """
    _ = args
    logging.info("Parse bb example")
    with tf.name_scope("DecodeProto/"):
        features = {
            "name": tf.FixedLenFeature([], tf.string),
            "data": tf.FixedLenFeature([], tf.string),
            "shape": tf.FixedLenFeature([2], tf.int64)
        }
        features = tf.parse_single_example(example_proto, features=features)
        data = tf.decode_raw(features["data"], tf.float32)
        cls_bb = tf.reshape(data, features["shape"])
    return {"cls_bb": cls_bb, "name": features["name"]}


def data_augmentation(features, labels, args):
    """ 2D
    c = args.input_group
    images: [h, w, c]
    labels: [h, w]    --> liver + tumor
            [h, w, c] --> only tumor
    """
    def _wrap_get_gd_image_multi_objs(mask_):
        guide_image = array_kits.get_gd_image_multi_objs(
            mask_,
            obj_value=2,
            center_perturb=args.center_perturb,
            stddev_perturb=args.stddev_perturb,
            partial=False,
            with_fake_guides=args.use_fake_guide,
            fake_rate=args.fake_rate,
            fake_range_value=1)[..., None]
        return guide_image.astype(np.float32)

    with tf.name_scope(PREPROCESS):
        liver = tf.ones_like(features["images"], features["images"].dtype)
        if args.only_tumor:
            # Assert images.shape == labels.shape
            liver = tf.clip_by_value(labels, 0, 1)
            labels = labels[..., args.input_group // 2]

        with tf.name_scope("Augmentation/"):
            if args.noise:
                # apply liver mask before flip and zoom
                features["images"] = image_ops.random_noise(features["images"], args.noise_scale)
                # Remove background
                features["images"] *= tf.cast(liver, features["images"].dtype)
                logging.info("Train: Add random noise, scale = {}".format(args.noise_scale))

            if args.flip:
                features["images"], labels = image_ops.random_flip_left_right(
                    features["images"], labels)
                logging.info("Train: Add random flip")

            if args.zoom:
                features["images"], labels = image_ops.random_zoom_in(features["images"], labels,
                                                                      args.zoom_scale)
                logging.info("Train: Add random zoom, scale = {}".format(args.zoom_scale))

            if args.use_spatial_guide:
                logging.info("Train: Add spatial guide")
                guide = tf.py_func(_wrap_get_gd_image_multi_objs, [labels], tf.float32)
                if args.random_spatial_guide:
                    mask = image_ops.random_zero_or_one(tf.shape(guide), guide.dtype)
                    guide = guide * mask
                    logging.info("Train: Add random spatial guide")
                features["images"] = tf.concat((features["images"], guide), axis=-1)

        # Resize only when batch size > 1 for batch operation
        if args.resize_for_batch and args.batch_size > 1:
            logging.info("Train: Resize image to {} x {}".format(args.im_height, args.im_width))
            image = tf.image.resize_bilinear(tf.expand_dims(features["images"], axis=0),
                                             [args.im_height, args.im_width])
            features["images"] = tf.squeeze(image, axis=0)
            logging.info("Train: Resize label to {} x {}".format(args.im_height, args.im_width))
            labels = tf.image.resize_nearest_neighbor(tf.expand_dims(tf.expand_dims(labels, axis=0), axis=-1),
                                                      [args.im_height, args.im_width])
            labels = tf.squeeze(labels, axis=(0, -1))
        if args.resize_for_batch and args.batch_size == 1:
            logging.warning("Train: Resize operation is skipped for batch_size=1. "
                            "Make sure that the shape of input images are aligned with multiplier "
                            "of 2**n where n is the pooling times if slim.conv2d_transpose is used.")

        if args.only_tumor:
            logging.info("Train: Clip label to [0, 1]")
            labels = tf.clip_by_value(labels - 1, 0, 1)

        # Bounding box: normalized coordinates --> actual coordinates
        if args.cls_branch:
            shape = tf.to_float(tf.shape(features["images"])[:-1])
            features["gt_boxes"] = tf.to_int32(tf.ceil(features["gt_boxes"] * (tf.expand_dims(
                tf.concat((shape, shape), axis=0), axis=0) - 1)))

    return features, labels


def data_processing_eval_while_train(features, labels, args):
    """ 2D
    c = args.input_group
    images: [h, w, c]
    labels: [h, w]    --> liver + tumor
            [h, w, c] --> only tumor
    """
    def _wrap_get_gd_image_multi_objs(mask_):
        guide_image = array_kits.get_gd_image_multi_objs(
            mask_,
            obj_value=2,
            center_perturb=args.center_perturb,
            stddev_perturb=args.stddev_perturb,
            partial=False,
            with_fake_guides=args.use_fake_guide,
            fake_rate=args.fake_rate,
            fake_range_value=1)[..., None]
        return guide_image.astype(np.float32)

    if args.only_tumor:
        # Assert images.shape == labels.shape
        labels = labels[..., args.input_group // 2]

    if args.use_spatial_guide:
        logging.info("Eval WT: Add spatial guide")
        guide = tf.py_func(_wrap_get_gd_image_multi_objs, [labels], tf.float32)
        features["images"] = tf.concat((features["images"], guide), axis=-1)

    # Resize only when batch size > 1 for batch operation
    if args.resize_for_batch and args.batch_size > 1:
        logging.info("Eval WT: Resize image to {} x {}".format(args.im_height, args.im_width))
        image = tf.image.resize_bilinear(
            tf.expand_dims(features["images"], axis=0),
            [args.im_height, args.im_width])
        features["images"] = tf.squeeze(image, axis=0)
        if not args.eval_3d:    # 3D evaluation don't need to resize label
            logging.info("Eval WT: Resize label to {} x {}".format(args.im_height, args.im_width))
            labels = tf.image.resize_nearest_neighbor(
                tf.expand_dims(tf.expand_dims(labels, axis=0), axis=-1),
                [args.im_height, args.im_width])
            labels = tf.squeeze(labels, axis=(0, -1))

    if args.only_tumor:
        logging.info("Eval WT: Clip label to [0, 1]")
        labels = tf.clip_by_value(labels - 1, 0, 1)

    # Bounding box: normalized coordinates --> actual coordinates
    if args.cls_branch:
        shape = tf.to_float(tf.shape(features["images"])[:-1])
        features["gt_boxes"] = tf.to_int32(tf.ceil(features["gt_boxes"] * (tf.expand_dims(
            tf.concat((shape, shape), axis=0), axis=0) - 1)))

    return features, labels


def data_processing_eval(features, labels, args):
    # Resize only when batch size > 1 for batch operation
    if args.resize_for_batch and args.batch_size > 1:
        logging.info("Eval: Resize image to {} x {}".format(args.im_height, args.im_width))
        image = tf.image.resize_bilinear(tf.expand_dims(features["images"], axis=0),
                                         [args.im_height, args.im_width])
        features["images"] = tf.squeeze(image, axis=0)
        # Label isn't need resize
    if args.resize_for_batch and args.batch_size == 1:
        logging.warning("Eval: Resize operation is skipped for batch_size=1. "
                        "Make sure that the shape of input images are aligned with multiplier "
                        "of 2**n where n is the pooling times if slim.conv2d_transpose is used.")

    if args.only_tumor:
        logging.info("Eval: Clip label to [0, 1]")
        # Add livers
        features["livers"] = tf.clip_by_value(labels, 0, 1)
        labels = tf.clip_by_value(labels - 1, 0, 1)

    return features, labels


def get_2d_multi_records_dataset_for_train(file_names, args):
    """
    Generate tf.data.Dataset from tf-record file for training

    Parameters
    ----------
    file_names: list or tuple
        A list of tf-record file names
    args: ArgumentParser
        Used arguments: batch_size, zoom, zoom_scale, noise, noise_scale, w_width, w_level

    Returns
    -------
    A tf.data.Dataset instance

    """
    parse_fn = partial(parse_2d_example_proto, args=args, mode=ModeKeys.TRAIN)
    augment_fn = partial(data_augmentation, args=args)
    filter_fn = partial(filter_slices, args=args, strategy="area", size=args.filter_size)

    if len(file_names) > 1:
        dataset = (Dataset.from_tensor_slices(file_names)
                   .shuffle(buffer_size=len(file_names), seed=SEED_FILE)
                   .interleave(lambda x: tf.data.TFRecordDataset(x),
                               cycle_length=len(file_names),
                               block_length=BLOCK_LENGTH))
    else:
        dataset = tf.data.TFRecordDataset(file_names[0])

    dataset = (dataset.filter(filter_fn)
               .apply(tf.data.experimental.shuffle_and_repeat(buffer_size=SHUFFLE_BUFFER_SIZE,
                                                              count=None, seed=SEED_BATCH))
               .map(parse_fn, num_parallel_calls=args.batch_size)
               .map(augment_fn, num_parallel_calls=args.batch_size)
               .batch(args.batch_size)
               .prefetch(buffer_size=args.batch_size))

    return dataset


def get_2d_multi_records_dataset_for_eval(file_names, args):
    parse_fn = partial(parse_2d_example_proto, args=args, mode="eval_while_train")
    augment_fn = partial(data_processing_eval_while_train, args=args)

    dataset = tf.data.TFRecordDataset(file_names[0])
    for file_name in file_names[1:]:
        dataset = dataset.concatenate(tf.data.TFRecordDataset(file_name))

    dataset = (dataset.map(parse_fn, num_parallel_calls=args.batch_size)
               .map(augment_fn, num_parallel_calls=args.batch_size)
               .batch(args.batch_size, drop_remainder=True)
               .prefetch(buffer_size=args.batch_size))

    return dataset


def _before_flat_fn(features, labels, args):
    """ 3D
    images: [d, h, w, c]
    labels: [d, h, w]

    Single channel with guide is deprecated!
    """
    def _wrap_get_gd_image_multi_objs(mask):
        guide_image = array_kits.get_gd_image_multi_objs(
            mask,
            obj_value=2,
            center_perturb=0.,
            stddev_perturb=0.,
            partial=True,
            with_fake_guides=False,
            partial_slice=args.guide)[..., None]
        return guide_image.astype(np.float32)

    if args.use_spatial_guide and args.use_fewer_guide:
        guide = tf.py_func(_wrap_get_gd_image_multi_objs, [labels], tf.float32)
        features["guides"] = guide
        if args.input_group == 1:
            features["images"] = tf.concat((features["images"], features.pop("guides")), axis=-1)
        else:
            # For multiple input channels, we do concat operation after constructing multiple channels
            pass

    return features, labels


def _flat_map_fn(x, y, mode, args):
    logging.info("Eval{}: Flap map".format(" WT" if mode == "eval_while_train" else ""))
    repeat_times = tf.shape(x["images"], out_type=tf.int64)[0]

    images = Dataset.from_tensor_slices(x["images"])
    labels = Dataset.from_tensor_slices(y)
    names = Dataset.from_tensors(x["names"]).repeat(repeat_times)
    pads = Dataset.from_tensors(x["pads"]).repeat(repeat_times)

    def map_fn(image, label, name, pad, *ex_args):
        features = {"images": image,
                    "names": name,
                    "pads": pad}
        if len(ex_args) == 0:
            return features, label
        elif len(ex_args) == 1:
            features["bboxes"] = ex_args[0]
            return features, label

    zip_list = [images, labels, names, pads]
    if args.resize_for_batch:
        zip_list.append(Dataset.from_tensors(x["bboxes"]).repeat(repeat_times))

    return Dataset.zip(tuple(zip_list)).map(map_fn)


def _flat_map_fn_multi_channels(x, y, mode, args):
    logging.info("Eval{}: Flap map for multi-channel image"
                 .format(" WT" if mode == "eval_while_train" else ""))
    repeat_times = tf.shape(x["images"], out_type=tf.int64)[0]

    n = args.input_group
    with tf.name_scope(PREPROCESS):
        shape = tf.concat(([n // 2], tf.shape(x["images"])[1:]), axis=0)
        zero_float = tf.zeros(shape, dtype=x["images"].dtype)
        new_images = tf.concat((zero_float, x["images"], zero_float), axis=0)
        concat_list = [new_images[x:-n + x + 1 if x < n - 1 else None] for x in range(n)]
        if "guides" in x:
            concat_list.append(x["guides"])
        image_multi_channels = tf.concat(concat_list, axis=-1)

    if args.guide != "middle":
        images = Dataset.from_tensor_slices(image_multi_channels)
        labels = Dataset.from_tensor_slices(y)
        names = Dataset.from_tensors(x["names"]).repeat(repeat_times)
        pads = Dataset.from_tensors(x["pads"]).repeat(repeat_times)
    else:   # "middle"
        images = Dataset.from_tensor_slices(tf.concat((image_multi_channels,
                                                       tf.reverse(image_multi_channels, [0])), axis=0))
        labels = Dataset.from_tensor_slices(tf.concat((y, tf.reverse(y, [0])), axis=0))
        names = Dataset.from_tensors(x["names"]).repeat(repeat_times)
        names_rev = Dataset.from_tensors(tf.string_join([x["names"],
                                                         tf.constant(".rev", tf.string)])).repeat(repeat_times)
        names = names.concatenate(names_rev)
        pads = Dataset.from_tensors(x["pads"]).repeat(repeat_times * 2)

    concat_list = (images, labels, names, pads)
    ex_arg_keys = []
    if args.resize_for_batch:
        bboxes = Dataset.from_tensors(x["bboxes"]).repeat(repeat_times * (1 if args.guide != "middle" else 2))
        concat_list += (bboxes,)
        ex_arg_keys.append("bboxes")
    if args.cls_branch:
        gt_boxes = Dataset.from_tensors(x["gt_boxes"]).repeat(repeat_times)
        im_info = Dataset.from_tensors(x["im_info"]).repeat(repeat_times)
        concat_list += (gt_boxes, im_info)
        ex_arg_keys.extend(["gt_boxes", "im_info"])

    def map_fn(image, label, name, pad, *ex_args):
        features = {"images": image,
                    "names": name,
                    "pads": pad}
        for i, ex_arg in enumerate(ex_args):
            features[ex_arg_keys[i]] = ex_arg
        return features, label

    return Dataset.zip(concat_list).map(map_fn)


def get_3d_multi_records_dataset_for_eval(file_names, mode, args):
    parse_fn = partial(parse_3d_example_proto, args=args)
    before_flat_fn = partial(_before_flat_fn, args=args)
    flat_fn = (partial(_flat_map_fn, mode=mode, args=args)
               if args.input_group == 1 else
               partial(_flat_map_fn_multi_channels, mode=mode, args=args))
    augment_fn = (partial(data_processing_eval_while_train, args=args)
                  if mode == "eval_while_train" else
                  partial(data_processing_eval, args=args))

    dataset = tf.data.TFRecordDataset(file_names[0])
    for file_name in file_names[1:]:
        dataset = dataset.concatenate(tf.data.TFRecordDataset(file_name))
    if args.eval_skip_num:
        dataset = dataset.skip(args.eval_skip_num)

    dataset = dataset.map(parse_fn, num_parallel_calls=args.batch_size)

    if args.use_fewer_guide:
        dataset = dataset.map(before_flat_fn, num_parallel_calls=args.batch_size)

    dataset = (dataset.flat_map(flat_fn)
               .map(augment_fn, num_parallel_calls=args.batch_size)
               .batch(args.batch_size)
               .prefetch(buffer_size=args.batch_size))

    return dataset


def get_2d_images_and_bboxes_dataset_for_train(image_filenames, bbox_filenames, args):
    augment_fn = partial(data_augmentation, args=args)
    filter_fn = partial(filter_slices, args=args, strategy="area", size=args.filter_size)

    def parse_img_box(*examples):
        features, labels = parse_2d_example_proto(examples[0], ModeKeys.TRAIN, args)
        bboxes = parse_bb_example_proto(examples[1], args)
        features["gt_boxes"] = bboxes["cls_bb"]
        # features["gt_name"] = bboxes["name"]
        return features, labels

    if len(image_filenames) != len(bbox_filenames):
        raise ValueError("Image and bbox shape mismatch: {} vs {}"
                         .format(len(image_filenames), len(bbox_filenames)))

    if len(image_filenames) > 1:
        dataset = (Dataset.from_tensor_slices(list(zip(image_filenames, bbox_filenames)))
                   .shuffle(buffer_size=len(image_filenames), seed=SEED_FILE)
                   .interleave(lambda x: (Dataset.zip((tf.data.TFRecordDataset(x[0]),    # For image/label
                                                       tf.data.TFRecordDataset(x[1])))),  # For bounding box
                               cycle_length=len(image_filenames),
                               block_length=BLOCK_LENGTH))
    else:
        dataset = Dataset.zip((tf.data.TFRecordDataset(image_filenames[0]),
                               tf.data.TFRecordDataset(bbox_filenames[0])))

    dataset = (dataset.filter(filter_fn)
               .apply(tf.data.experimental.shuffle_and_repeat(buffer_size=SHUFFLE_BUFFER_SIZE,
                                                              count=None, seed=SEED_BATCH))
               .map(parse_img_box, num_parallel_calls=args.batch_size)
               .map(augment_fn, num_parallel_calls=args.batch_size)
               .batch(1)
               .prefetch(buffer_size=8))

    return dataset


def get_2d_images_and_bboxes_dataset_for_eval(image_filenames, bbox_filenames, args):
    augment_fn = partial(data_processing_eval_while_train, args=args)

    def parse_img_box(*examples):
        features, labels = parse_2d_example_proto(examples[0], args=args, mode="eval_while_train")
        bboxes = parse_bb_example_proto(examples[1], args)
        features["gt_boxes"] = bboxes["cls_bb"]
        # features["gt_name"] = bboxes["name"]
        return features, labels

    if len(image_filenames) != len(bbox_filenames):
        raise ValueError("Image and bbox shape mismatch: {} vs {}"
                         .format(len(image_filenames), len(bbox_filenames)))

    dataset = Dataset.zip((tf.data.TFRecordDataset(image_filenames[0]),
                           tf.data.TFRecordDataset(bbox_filenames[0])))
    for image_filename, bbox_filename in zip(image_filenames[1:], bbox_filenames[1:]):
        dataset = dataset.concatenate(Dataset.zip((tf.data.TFRecordDataset(image_filename),
                                                   tf.data.TFRecordDataset(bbox_filename))))

    dataset = (dataset.map(parse_img_box, num_parallel_calls=args.batch_size)
               .map(augment_fn, num_parallel_calls=args.batch_size)
               .batch(1)
               .prefetch(buffer_size=8))

    return dataset


def get_3d_images_and_bboxes_dataset_for_eval(image_filenames, args):
    flat_fn = partial(_flat_map_fn_multi_channels, args=args)
    augment_fn = partial(data_processing_eval, args=args)

    def parse_img_box(*examples):
        features, labels = parse_3d_example_proto(examples[0], args=args)
        features["gt_boxes"] = examples[1]
        return features, labels

    dataset = Dataset.zip((tf.data.TFRecordDataset(image_filenames[0]),
                           Dataset.from_tensors([[0.] * 4]).repeat(None)))
    for image_filename in image_filenames[1:]:
        dataset = dataset.concatenate(Dataset.zip((tf.data.TFRecordDataset(image_filename),
                                                   Dataset.from_tensor_slices([[0.] * 4]).repeat(None))))
    if args.eval_skip_num:
        dataset = dataset.skip(args.eval_skip_num)

    dataset = (dataset.map(parse_img_box, num_parallel_calls=args.batch_size)
               .flat_map(flat_fn)
               .map(augment_fn, num_parallel_calls=args.batch_size)
               .batch(1)
               .prefetch(buffer_size=8))

    return dataset
