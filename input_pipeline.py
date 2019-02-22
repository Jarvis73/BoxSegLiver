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

import tensorflow as tf
from functools import partial
from pathlib import Path
from tensorflow.python.platform import tf_logging as logging

from utils import image_ops

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
PREPROCESS = "Preprocess"


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
    group.add_argument("--w_width",
                       type=float,
                       default=450,
                       required=False, help="Medical image window width (default: %(default)f)")
    group.add_argument("--w_level",
                       type=float,
                       default=50,
                       required=False, help="Medical image window level (default: %(default)f)")
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


def _collect_datasets(datasets):
    tf_records = []

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
            if len(tf_records) > 1:
                return get_multi_records_dataset_for_train(tf_records, args)
            elif len(tf_records) == 1:
                return get_record_dataset_for_train(tf_records[0], args)
            else:
                raise ValueError("No valid dataset found for training")
        elif mode == ModeKeys.EVAL or mode == ModeKeys.PREDICT:
            tf_records = _collect_datasets(args.dataset_for_eval)
            if len(tf_records) > 1:
                return get_multi_records_dataset_for_eval(tf_records, args)
            elif len(tf_records) == 1:
                return get_record_dataset_for_eval(tf_records[0], args)
            else:
                raise ValueError("No valid dataset found for evaluation")


def filter_slices(example_proto, strategy="empty", **kwargs):
    """
    Filter slices with specified strategy

    Parameters
    ----------
    example_proto: protobuf
        TF-Examaple
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
        if kwargs["size"] < 1:
            raise ValueError("Argument size must be greater than 1, got {}".format(kwargs["size"]))

    with tf.name_scope("DecodeProto"):
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
            num_dense = tf.count_nonzero(label)
            return tf.greater_equal(num_dense, kwargs["size"])


def parse_2d_example_proto(example_proto, args):
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
    with tf.name_scope("DecodeProto"):
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

        image = tf.decode_raw(features["image/encoded"], tf.int16, name="DecodeImage")
        image = tf.reshape(image, features["image/shape"], name="ReshapeImage")
        image = tf.to_float(image)
        label = tf.decode_raw(features["segmentation/encoded"], tf.uint8, name="DecodeMask")
        label = tf.reshape(label, features["segmentation/shape"], name="ReshapeMask")
        label = tf.to_int32(label)
        if len(args.classes) == 1:  # Only liver
            label = tf.clip_by_value(label, 0, 1)

    with tf.name_scope(PREPROCESS):
        image = image_ops.random_adjust_window_width_level(image, args.w_width, args.w_level,
                                                           SEED_WIDTH, SEED_LEVEL)

    # return features and labels
    return {"images": image,
            "name": features["image/name"],
            "id": features["extra/number"]}, label


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
    with tf.name_scope("DecodeProto"):
        features = tf.parse_single_example(
            example_proto,
            features={
                "image/encoded": tf.FixedLenFeature([], tf.string),
                "image/shape": tf.FixedLenFeature([4], tf.int64),
                "image/name": tf.FixedLenFeature([], tf.string),
                "segmentation/encoded": tf.FixedLenFeature([], tf.string),
                "segmentation/shape": tf.FixedLenFeature([3], tf.int64),
            }
        )

        image = tf.decode_raw(features["image/encoded"], tf.int16, name="DecodeImage")
        image = tf.reshape(image, features["image/shape"], name="ReshapeImage")
        image = tf.to_float(image)
        label = tf.decode_raw(features["segmentation/encoded"], tf.uint8, name="DecodeMask")
        label = tf.reshape(label, features["segmentation/shape"], name="ReshapeMask")
        label = tf.to_int32(label)
        if len(args.classes) == 1:  # Only liver
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

    return {"images": image,
            "name": features["image/name"],
            "pad": pad}, label


def data_augmentation(features, labels, args):
    """
    Perform data augmentation

    Parameters
    ----------
    features
    labels
    args: ArgumentParser
        Used arguments: zoom, zoom_scale, noise, noise_scale

    Returns
    -------
    features and labels

    """
    with tf.name_scope(PREPROCESS):
        with tf.name_scope("Augmentation"):
            # Random zoom in
            if args.zoom:
                features["images"], labels = image_ops.random_zoom_in(features["images"], labels,
                                                                      args.zoom_scale)
                logging.info("Add random zoom, scale = {}".format(args.zoom_scale))

            if args.noise:
                features["images"] = image_ops.random_noise(features["images"], args.noise_scale)
                logging.info("Add random noise, scale = {}".format(args.noise_scale))

            return features, labels


def get_record_dataset_for_train(file_name, args):
    """
    Generate tf.data.Dataset from tf-record file for training

    Parameters
    ----------
    file_name
    args: ArgumentParser
        Used arguments: batch_size, zoom, zoom_scale, noise, noise_scale, w_width, w_level

    Returns
    -------

    """
    parse_fn = partial(parse_2d_example_proto, args=args)
    augment_fn = partial(data_augmentation, args=args)
    filter_fn = partial(filter_slices, strategy="area", size=100)

    dataset = (tf.data.TFRecordDataset(file_name)
               .filter(filter_fn)
               .apply(tf.data.experimental.shuffle_and_repeat(buffer_size=SHUFFLE_BUFFER_SIZE, count=None,
                                                              seed=SEED_BATCH))
               .map(parse_fn, num_parallel_calls=args.batch_size)
               .map(augment_fn, num_parallel_calls=args.batch_size)
               .batch(args.batch_size)
               .prefetch(buffer_size=args.batch_size))    # for acceleration,

    return dataset


def get_multi_records_dataset_for_train(file_names, args):
    parse_fn = partial(parse_2d_example_proto, args=args)
    augment_fn = partial(data_augmentation, args=args)
    filter_fn = partial(filter_slices, strategy="area", size=100)

    dataset = (Dataset.from_tensor_slices(file_names)
               .shuffle(buffer_size=len(file_names), seed=SEED_FILE)
               .interleave(lambda x: (tf.data.TFRecordDataset(x)
                                      .filter(filter_fn)),
                           cycle_length=len(file_names),
                           block_length=BLOCK_LENGTH)
               .apply(tf.data.experimental.shuffle_and_repeat(buffer_size=SHUFFLE_BUFFER_SIZE, count=None,
                                                              seed=SEED_BATCH))
               .map(parse_fn, num_parallel_calls=args.batch_size)
               .map(augment_fn, num_parallel_calls=args.batch_size)
               .batch(args.batch_size)
               .prefetch(buffer_size=args.batch_size))

    return dataset


def _flat_map_fn(x, y):
    repeat_times = tf.shape(x["images"], out_type=tf.int64)[0]

    images = Dataset.from_tensor_slices(x["images"]),
    labels = Dataset.from_tensor_slices(y),
    names = Dataset.from_tensors(x["name"]).repeat(repeat_times),
    pads = Dataset.from_tensors(x["pad"]).repeat(repeat_times)

    def map_fn(image, label, name, pad):
        return {"images": image[0],
                "names": name[0],
                "pads": pad,
                "labels": label[0]}

    return Dataset.zip((images, labels, names, pads)).map(map_fn)


def get_record_dataset_for_eval(file_name, args):
    parse_fn = partial(parse_3d_example_proto, args=args)

    # Read 3d dataset and convert to 2d dataset
    dataset = (tf.data.TFRecordDataset(file_name)
               .map(parse_fn)
               .flat_map(_flat_map_fn)
               .batch(args.batch_size)
               .prefetch(buffer_size=args.batch_size))

    return dataset


def get_multi_records_dataset_for_eval(file_names, args):
    parse_fn = partial(parse_3d_example_proto, args=args)

    dataset = (Dataset.from_tensor_slices(file_names)
               .interleave(lambda x: (tf.data.TFRecordDataset(x)),
                           cycle_length=len(file_names),
                           block_length=BLOCK_LENGTH)
               .map(parse_fn, num_parallel_calls=args.batch_size)
               .flat_map(_flat_map_fn)
               .batch(args.batch_size)
               .prefetch(buffer_size=args.batch_size))

    return dataset


# def main():
#     import argparse
#     import models
#     import matplotlib.pyplot as plt
#     parser = argparse.ArgumentParser()
#     add_arguments(parser)
#     models.add_arguments(parser)
#     args = parser.parse_args()
#     logging.set_verbosity(logging.INFO)
#
#     file_names = [r"D:\documents\MLearning\MultiOrganDetection\core\MedicalImageSegmentation\data\LiTS\records\test-2D-3-of-5.tfrecord",
#                   r"D:\documents\MLearning\MultiOrganDetection\core\MedicalImageSegmentation\data\LiTS\records\test-2D-4-of-5.tfrecord"]
#     dataset = get_multi_records_dataset_for_train(file_names, args)
#
#     features, labels = dataset.make_one_shot_iterator().get_next()
#
#     sess = tf.Session()
#
#     while True:
#         img, lab = sess.run([features["images"], labels])
#         plt.subplot(121)
#         plt.imshow(img[0, ..., 0], cmap="gray")
#         plt.subplot(122)
#         plt.imshow(lab[0], cmap="gray")
#         plt.show()
#
#
# if __name__ == "__main__":
#     main()
