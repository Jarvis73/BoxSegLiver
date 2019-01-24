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

from utils import image_ops

Dataset = tf.data.Dataset

NUM_PARALLEL_READ = 4

# number for debug, None for training
SEED_FILE = 3456
SEED_BATCH = 7890
SEED_WIDTH = 3489
SEED_LEVEL = 8062

# record block length
BLOCK_LENGTH = 16


def filter_lits_liver(example_proto):
    with tf.name_scope("FilterEmpty"):
        features = tf.parse_single_example(
            example_proto,
            features={
                "extra/empty": tf.FixedLenFeature([], tf.int64)
            }
        )

        return tf.equal(features["extra/empty"], 0)


def parse_lits_liver(example_proto, args):
    with tf.name_scope("DecodeProto"):
        features = tf.parse_single_example(
            example_proto,
            features={
                "image/encoded": tf.FixedLenFeature([], tf.string),
                "image/shape": tf.FixedLenFeature([], tf.int64),
                "segmentation/encoded": tf.FixedLenFeature([], tf.string),
                "segmentation/shape": tf.FixedLenFeature([], tf.int64),
                "extra/number": tf.FixedLenFeature([], tf.int64),
            }
        )

        image = tf.decode_raw(features["image/encoded"], tf.int16, name="DecodeImage")
        image = tf.reshape(image, features["image/shape"], name="ReshapeImage")
        image = tf.to_float(image)
        label = tf.decode_raw(features["segmentation/encoded"], tf.uint8, name="DecodeMask")
        label = tf.reshape(label, features["segmentation/shape"], name="ReshapeMask")
        label = tf.to_int32(label)

    if args.mode == "TRAIN":
        image = image_ops.random_adjust_window_width_level(image, args.w_width, args.w_level,
                                                           SEED_WIDTH, SEED_LEVEL)
    elif args.mode in ["EVAL", "PREDICT"]:
        image = image_ops.adjust_window_width_level(image, args.w_width, args.w_level)
    else:
        raise ValueError("Not supported mode: " + args.mode)

    return image, label


def data_augmentation(examples, args):
    with tf.name_scope("Augmentation"):
        image, label = examples[0], examples[1]
        # Random zoom in
        if args.zoom:
            image, label = image_ops.random_zoom_in(image, label, args.zoom_scale)
        if args.noise:
            image = image_ops.random_noise(image, args.noise_scale)

        return image, label


def get_lits_liver_dataset_for_train(file_names, args):
    dataset = (Dataset.from_tensor_slices(file_names)
               .shuffle(buffer_size=len(file_names), seed=SEED_FILE)
               .interleave(lambda x: (tf.data.TFRecordDataset(x)
                                      .filter(filter_lits_liver)),
                           cycle_length=len(file_names),
                           block_length=BLOCK_LENGTH)
               .apply(tf.data.experimental.shuffle_and_repeat(buffer_size=160, count=None, seed=SEED_BATCH))
               .map(partial(parse_lits_liver, args=args), num_parallel_calls=args.batch_size)
               .map(partial(data_augmentation, args=args), num_parallel_calls=args.batch_size)
               .batch(args.batch_size)
               .prefetch(buffer_size=args.batch_size)    # for acceleration,
               .make_one_shot_iterator())

    return dataset.get_next()

