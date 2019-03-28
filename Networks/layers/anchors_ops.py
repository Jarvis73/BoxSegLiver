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

"""
References: https://github.com/fizyr/keras-retinanet

Modified by: Jianwei Zhang
"""

import numpy as np
import tensorflow as tf


def generate_anchors(base_size=16, ratios=None, scales=None):
    num_anchors = len(ratios) * len(scales)
    # initialize output anchors
    anchors = np.zeros((num_anchors, 4))
    # scale base_size
    anchors[:, 2:] = base_size * np.tile(scales, (2, len(ratios))).T
    # compute areas of anchors
    areas = anchors[:, 2] * anchors[:, 3]

    # correct for ratios
    anchors[:, 2] = np.sqrt(areas / np.repeat(ratios, len(scales)))
    anchors[:, 3] = anchors[:, 2] * np.repeat(ratios, len(scales))

    # transform from (x_ctr, y_ctr, w, h) -> (x1, y1, x2, y2)
    anchors[:, 0::2] -= np.tile(anchors[:, 2] * 0.5, (2, 1)).T
    anchors[:, 1::2] -= np.tile(anchors[:, 3] * 0.5, (2, 1)).T

    return anchors.astype(np.float32)


def shift(shape, stride, anchors):
    """ Produce shifted anchors based on shape of the map and stride size.

    Args
        shape  : Shape to shift the anchors over.
        stride : Stride to shift the anchors with over the shape.
        anchors: The anchors to apply at each location.
    """
    shift_x = (tf.range(0, tf.to_float(shape[1]), dtype=tf.float32) + tf.constant(0.5, dtype=tf.float32)) * tf.to_float(stride)
    shift_y = (tf.range(0, tf.to_float(shape[0]), dtype=tf.float32) + tf.constant(0.5, dtype=tf.float32)) * tf.to_float(stride)

    shift_x, shift_y = tf.meshgrid(shift_x, shift_y)
    shift_x = tf.reshape(shift_x, [-1])
    shift_y = tf.reshape(shift_y, [-1])

    shifts = tf.stack([shift_x, shift_y, shift_x, shift_y], axis=0)

    shifts = tf.transpose(shifts)
    A = tf.shape(anchors)[0]

    K = tf.shape(shifts)[0]  # number of base points = feat_h * feat_w

    shifted_anchors = tf.reshape(anchors, [1, A, 4]) + tf.cast(tf.reshape(shifts, [K, 1, 4]), tf.float32)
    shifted_anchors = tf.reshape(shifted_anchors, [K * A, 4])

    return shifted_anchors


if __name__ == "__main__":
    a = generate_anchors(8, [0.5, 1, 2], [2 ** (x / 3) for x in [1, 2, 3]])
    print(a[:, 2:4] - a[:, 0:2])
    a = generate_anchors(16, [0.5, 1, 2], [2 ** (x / 3) for x in [1, 2, 3]])
    print(a[:, 2:4] - a[:, 0:2])
    a = generate_anchors(32, [0.5, 1, 2], [2 ** (x / 3) for x in [1, 2, 3]])
    print(a[:, 2:4] - a[:, 0:2])
