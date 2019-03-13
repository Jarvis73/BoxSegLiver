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
    group.add_argument("--dataset_for_eval",
                       type=str,
                       nargs="*",
                       required=False, help="TFRecord names for model input. Such as "
                                            "\"LiTS/records/data_fold_0.tfrecord\" or a *.json file "
                                            "(Check data/dataset_example.json for details)")
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
    group.add_argument("--triplet",
                       action="store_true",
                       required=False, help="Use triplet as inputs")
    group.add_argument("--quintuplet",
                       action="store_true",
                       required=False, help="Use quintuplet as inputs")
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

    with tf.variable_scope("InputPipeline"):
        if mode == ModeKeys.TRAIN:
            tf_records = _collect_datasets(args.dataset_for_train)
            if len(tf_records) >= 1:
                # if args.triplet:
                #     return get_multi_records_dataset_for_train(tf_records, args)
                return get_multi_records_dataset_for_train(tf_records, args)
            else:
                raise ValueError("No valid dataset found for training")
        elif mode == ModeKeys.EVAL or mode == ModeKeys.PREDICT:
            tf_records = _collect_datasets(args.dataset_for_eval)
            if len(tf_records) >= 1:
                if args.triplet:
                    return get_multi_channels_dataset_for_eval(tf_records, args)
                return get_multi_records_dataset_for_eval(tf_records, args)
            else:
                raise ValueError("No valid dataset found for evaluation")


def filter_slices(example_proto, args, strategy="empty", **kwargs):
    """
    Filter slices with specified strategy

    Parameters
    ----------
    example_proto: protobuf
        TF-Examaple
    args: ArgumentParser
        Used arguments: w_width, w_level
    strategy: str
        Support `empty` and `area`
    kwargs: dict
        If strategy is `area`, then keyword argument `size` is necessary as a threshold.

    Returns
    -------

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
            features = tf.parse_single_example(
                example_proto,
                features={
                    "extra/empty": tf.FixedLenFeature([], tf.int64)
                }
            )

            return tf.equal(features["extra/empty"], 0)
        else:   # strategy == "area"
            features = tf.parse_single_example(
                example_proto,
                features={
                    "segmentation/encoded": tf.FixedLenFeature([], tf.string)
                }
            )
            label = tf.decode_raw(features["segmentation/encoded"], tf.uint8, name="DecodeMask")
            if args.only_tumor:
                obj_label = tf.clip_by_value(tf.to_int32(label) - 1, 0, 1)
            else:
                obj_label = label
            num_dense = tf.count_nonzero(obj_label)
            return tf.greater_equal(num_dense, kwargs["size"])


def parse_2d_example_proto(example_proto, args):
    """
    Parse example proto

    Parameters
    ----------
    example_proto: protobuf
        TF-Examaple
    args: ArgumentParser
        Used arguments: w_width, w_level

    Returns
    -------
    features and labels

    """
    with tf.name_scope("DecodeProto/"):
        features = tf.parse_single_example(
            example_proto,
            features={
                "image/encoded": tf.FixedLenFeature([], tf.string),
                "image/shape": tf.FixedLenFeature([3], tf.int64),
                "image/name": tf.FixedLenFeature([], tf.string),
                "segmentation/encoded": tf.FixedLenFeature([], tf.string),
                "segmentation/shape": tf.FixedLenFeature([2], tf.int64),
                "extra/number": tf.FixedLenFeature([], tf.int64),
            }
        )

        with tf.name_scope(PREPROCESS):
            image = tf.decode_raw(features["image/encoded"], tf.int16, name="DecodeImage")
            image = tf.reshape(image, features["image/shape"], name="ReshapeImage")
            image = tf.to_float(image)
            label = tf.decode_raw(features["segmentation/encoded"], tf.uint8, name="DecodeMask")
            label = tf.reshape(label, features["segmentation/shape"], name="ReshapeMask")
            label = tf.to_int32(label)
            if len(args.classes) == 1 and not args.only_tumor:  # Only liver
                label = tf.clip_by_value(label, 0, 1)

            image = image_ops.random_adjust_window_width_level(image, args.w_width, args.w_level,
                                                               SEED_WIDTH, SEED_LEVEL)

            ret_features = {"images": image,
                            "names": features["image/name"]}

            if hasattr(args, "train_without_eval") and not args.train_without_eval:
                ret_features["pads"] = tf.constant(0, dtype=tf.int64)
            if args.resize_for_batch:
                ret_features["bboxes"] = tf.constant([0] * 6, dtype=tf.int64)

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

        ret_features = {"images": image,
                        "names": features["image/name"],
                        "pads": pad}

        if args.resize_for_batch:
            ret_features["bboxes"] = features["extra/bbox"]

    return ret_features, label


def data_augmentation(features, labels, args, mode, only_first_slice=False):
    """
    Perform data augmentation

    Parameters
    ----------
    features
    labels
    args: ArgumentParser
        Used arguments: zoom, zoom_scale, noise, noise_scale
    mode:
    only_first_slice:

    Returns
    -------
    features and labels

    """
    def _wrap_get_gd_image_multi_objs(mask):
        center_perturb = args.center_perturb if args.mode == ModeKeys.TRAIN else 0.
        stddev_perturb = args.stddev_perturb if args.mode == ModeKeys.TRAIN else 0.
        with_fake = args.use_fake_guide if mode == ModeKeys.TRAIN else False
        guide_image = array_kits.get_gd_image_multi_objs(
            mask,
            obj_value=2,
            center_perturb=center_perturb,
            stddev_perturb=stddev_perturb,
            partial=only_first_slice,
            with_fake_guides=with_fake,
            fake_rate=args.fake_rate,
            fake_range_value=1)[..., None]
        return guide_image.astype(np.float32)

    with tf.name_scope(PREPROCESS):
        if mode == ModeKeys.TRAIN:
            with tf.name_scope("Augmentation/"):
                if args.flip:
                    features["images"], labels = image_ops.random_flip_left_right(
                        features["images"], labels)
                    logging.info("Add random flip")

                if args.zoom:
                    features["images"], labels = image_ops.random_zoom_in(features["images"], labels,
                                                                          args.zoom_scale)
                    logging.info("Add random zoom, scale = {}".format(args.zoom_scale))

                if args.noise:
                    features["images"] = image_ops.random_noise(features["images"], args.noise_scale)
                    logging.info("Add random noise, scale = {}".format(args.noise_scale))

        if args.use_spatial_guide and not args.use_fewer_guide:
            guide = tf.py_func(_wrap_get_gd_image_multi_objs, [labels], tf.float32)
            if args.use_fake_guide:
                logging.info("Add fake spatial guide")
            if mode == ModeKeys.TRAIN and args.random_spatial_guide:
                with tf.name_scope("Augmentation/"):
                    mask = image_ops.random_zero_or_one(tf.shape(guide), guide.dtype)
                    guide = guide * mask
                    logging.info("Add random spatial guide")
            features["images"] = tf.concat((features["images"], guide), axis=-1)

        if args.resize_for_batch:
            image = tf.image.resize_bilinear(tf.expand_dims(features["images"], axis=0),
                                             [args.im_height, args.im_width])
            features["images"] = tf.squeeze(image, axis=0)
            if args.only_tumor:  # Keep liver for loss mask
                features["livers"] = tf.clip_by_value(labels, 0, 1)
                features["livers"] = tf.image.resize_nearest_neighbor(tf.expand_dims(
                    tf.expand_dims(features["livers"], axis=0), axis=-1),
                    [args.im_height, args.im_width])
                features["livers"] = tf.squeeze(features["livers"], axis=(0, -1))
                labels = tf.clip_by_value(labels - 1, 0, 1)
            if mode == ModeKeys.TRAIN:
                labels = tf.image.resize_nearest_neighbor(tf.expand_dims(tf.expand_dims(labels, axis=0), axis=-1),
                                                          [args.im_height, args.im_width])
                labels = tf.squeeze(labels, axis=(0, -1))

    return features, labels


def get_multi_records_dataset_for_train(file_names, args):
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
    parse_fn = partial(parse_2d_example_proto, args=args)
    augment_fn = partial(data_augmentation, args=args, mode=ModeKeys.TRAIN)
    filter_fn = partial(filter_slices, args=args, strategy="area", size=args.filter_size)

    if len(file_names) > 1:
        dataset = (Dataset.from_tensor_slices(file_names)
                   .shuffle(buffer_size=len(file_names), seed=SEED_FILE)
                   .interleave(lambda x: (tf.data.TFRecordDataset(x)
                                          .filter(filter_fn)),
                               cycle_length=len(file_names),
                               block_length=BLOCK_LENGTH))
    else:
        dataset = tf.data.TFRecordDataset(file_names[0]).filter(filter_fn)

    dataset = (dataset.apply(tf.data.experimental.shuffle_and_repeat(buffer_size=SHUFFLE_BUFFER_SIZE,
                                                                     count=None, seed=SEED_BATCH))
               .map(parse_fn, num_parallel_calls=args.batch_size)
               .map(augment_fn, num_parallel_calls=args.batch_size)
               .batch(args.batch_size)
               .prefetch(buffer_size=args.batch_size))

    return dataset


def _flat_map_fn(x, y, args):
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


def _before_flat_fn(features, labels, args):

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

    if args.mode == ModeKeys.EVAL:
        if args.use_spatial_guide and args.use_fewer_guide:
            guide = tf.py_func(_wrap_get_gd_image_multi_objs, [labels], tf.float32)
            features["guides"] = guide
            if not args.triplet:
                features["images"] = tf.concat((features["images"], features.pop("guides")), axis=-1)

    return features, labels


def get_multi_records_dataset_for_eval(file_names, args):
    parse_fn = partial(parse_3d_example_proto, args=args)
    before_flat_fn = partial(_before_flat_fn, args=args)
    flat_fn = partial(_flat_map_fn, args=args)
    augment_fn = partial(data_augmentation, args=args, mode=ModeKeys.EVAL)

    dataset = tf.data.TFRecordDataset(file_names[0])
    for file_name in file_names[1:]:
        dataset = dataset.concatenate(tf.data.TFRecordDataset(file_name))
    if args.eval_skip_num:
        dataset = dataset.skip(args.eval_skip_num)

    dataset = (dataset.map(parse_fn, num_parallel_calls=args.batch_size)
               .map(before_flat_fn, num_parallel_calls=args.batch_size)
               .flat_map(flat_fn)
               .map(augment_fn, num_parallel_calls=args.batch_size)
               .batch(args.batch_size)
               .prefetch(buffer_size=args.batch_size))

    return dataset


# TODO(ZJW) Deprecated
def merge_to_channels(f1, f2, f3):
    logging.warning("Deprecated function! Use `get_multi_records_dataset_for_train`.")
    features = {}
    with tf.name_scope(PREPROCESS):
        features["images"] = tf.concat((f1[0]["images"], f2[0]["images"], f3[0]["images"]), axis=-1)

    for key, value in f1[0].items():
        if key == "images":
            continue
        features[key] = value

    return features, f2[1]


# TODO(ZJW) Deprecated
def get_multi_channels_dataset_for_train(file_names, args):
    logging.warning("Deprecated function! Use `get_multi_records_dataset_for_train`.")
    parse_fn = partial(parse_2d_example_proto, args=args)
    augment_fn = partial(data_augmentation, args=args, mode=ModeKeys.TRAIN)
    filter_fn = partial(filter_slices, args=args, strategy="area", size=args.filter_size)

    def filter_mismatching_elements(f1, f2, f3):
        with tf.name_scope(PREPROCESS):
            return tf.logical_and(tf.equal(f1[0]["names"], f2[0]["names"]),
                                  tf.equal(f2[0]["names"], f3[0]["names"]))

    def filter_triple_slices(f1, f2, f3):
        del f1, f3
        with tf.name_scope(PREPROCESS):
            return tf.greater_equal(tf.count_nonzero(f2[1]), args.filter_size)

    if len(file_names) > 1:
        dataset = (Dataset.from_tensor_slices(file_names)
                   .shuffle(buffer_size=len(file_names), seed=SEED_FILE)
                   .interleave(lambda x: (tf.data.TFRecordDataset(x)
                                          .filter(filter_fn)),
                               cycle_length=len(file_names),
                               block_length=BLOCK_LENGTH)
                   .map(parse_fn, num_parallel_calls=args.batch_size))
    else:
        dataset = (tf.data.TFRecordDataset(file_names[0])
                   .map(parse_fn, num_parallel_calls=args.batch_size))

    dataset = (Dataset.zip((dataset, dataset.skip(1), dataset.skip(2)))
               .filter(filter_triple_slices)
               .filter(filter_mismatching_elements)
               .apply(tf.data.experimental.shuffle_and_repeat(buffer_size=SHUFFLE_BUFFER_SIZE,
                                                              count=None, seed=SEED_BATCH))
               .map(merge_to_channels, num_parallel_calls=args.batch_size)
               .map(augment_fn, num_parallel_calls=args.batch_size)
               .batch(args.batch_size)
               .prefetch(buffer_size=args.batch_size))

    return dataset


def _flat_map_fn_multi_channels(x, y, args):
    repeat_times = tf.shape(x["images"], out_type=tf.int64)[0]

    with tf.name_scope(PREPROCESS):
        shape = tf.concat(([1], tf.shape(x["images"])[1:]), axis=0)
        zero_float = tf.zeros(shape, dtype=x["images"].dtype)
        new_images = tf.concat((zero_float, x["images"], zero_float), axis=0)
        if "guides" in x:
            concat_list = (new_images[:-2], x["images"], new_images[2:], x["guides"])
        else:
            concat_list = (new_images[:-2], x["images"], new_images[2:])
        image_multi_channels = tf.concat(concat_list, axis=-1)

    if args.guide == "first":
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

    def map_fn(image, label, name, pad, *ex_args):
        features = {"images": image,
                    "names": name,
                    "pads": pad}
        if len(ex_args) == 0:
            return features, label
        elif len(ex_args) == 1:
            features["bboxes"] = ex_args[0]
            return features, label

    if args.resize_for_batch:
        bboxes = Dataset.from_tensors(x["bboxes"]).repeat(repeat_times * (1 if args.guide == "first" else 2))
        return Dataset.zip((images, labels, names, pads, bboxes)).map(map_fn)

    return Dataset.zip((images, labels, names, pads)).map(map_fn)


def get_multi_channels_dataset_for_eval(file_names, args):
    parse_fn = partial(parse_3d_example_proto, args=args)
    before_flat_fn = partial(_before_flat_fn, args=args)
    flat_fn = partial(_flat_map_fn_multi_channels, args=args)
    augment_fn = partial(data_augmentation, args=args, mode=ModeKeys.EVAL)

    dataset = tf.data.TFRecordDataset(file_names[0])
    for file_name in file_names[1:]:
        dataset = dataset.concatenate(tf.data.TFRecordDataset(file_name))

    dataset = dataset.map(parse_fn, num_parallel_calls=args.batch_size)

    if args.use_fewer_guide:
        dataset = dataset.map(before_flat_fn, num_parallel_calls=args.batch_size)

    dataset = (dataset.flat_map(flat_fn)
               .map(augment_fn, num_parallel_calls=args.batch_size)
               .batch(args.batch_size)
               .prefetch(buffer_size=args.batch_size))

    return dataset
