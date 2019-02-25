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
Some utility functions run in tensorflow graph
"""

import random
import tensorflow as tf


def adjust_window_width_level(image, w_width, w_level, name=None):
    """
    Adjust the window width and window level of a gray-scale image.

    Typically, this function is useful to 16bit medical image, converting to
    a float32 image whose pixel values are in [0, 1] with proper window width and level.

    Parameters
    ----------
    image: Tensor
        4-D Tensor of shape [batch, height, width, channels]
    w_width: Scalar
        window width
    w_level: Scalar
        window level
    name: str
        operation name

    Returns
    -------
    A window-width-level-adjusted tensor of the same shape as `image` with float32 type.

    """
    with tf.name_scope(name, 'adjust_window_width_level', [image, w_width, w_level]):
        image = tf.convert_to_tensor(image, name="image")
        flt_image = tf.image.convert_image_dtype(image, tf.float32)

        lower, upper = w_level - w_width / 2, w_level + w_width / 2
        adjusted = (tf.clip_by_value(flt_image, lower, upper) - lower) / w_width

        return adjusted


def random_adjust_window_width_level(image, w_width, w_level, seed1=None, seed2=None, name=None):
    """
    Random adjust the window width and window level of a gray-scale image.

    Typically, this function is useful to 16bit medical image, converting to
    a float32 image whose pixel values are in [0, 1] with proper window width and level.

    Parameters
    ----------
    image: Tensor
        4-D Tensor of shape [batch, height, width, channels]
    w_width: Scalar
        window width
    w_level: Scalar
        window level
    seed1: Scalar
        random seed for window width
    seed2: Scalar
        random seed for window level
    name: str
        operation name

    Returns
    -------
    A randomly window-width-level-adjusted tensor of the same shape as `image` with float32 type.

    """
    with tf.name_scope(None, 'random_adjust_wwl', [image, w_width, w_level, seed1, seed2]):
        rd_width = tf.random_uniform([], -50, 50, seed=seed1)
        rd_level = tf.random_uniform([], -15, 15, seed=seed2)
        new_width = tf.add(float(w_width), rd_width, name="rw_width")
        new_level = tf.add(float(w_level), rd_level, name="rw_level")
        adjusted = adjust_window_width_level(image, new_width, new_level)

        return adjusted


def random_zoom_in(image, label=None, max_scale=1.5, seed_scale=None, seed_shift=None, name=None):
    """
    Randomly scale and shift an image.

    Parameters
    ----------
    image: Tensor
        images to zoom in. 3-D [height, width, channels] or 4-D [batch_size, height, width, channels]
    label: Tensor
        labels to zoom in. 2-D [height, width] or 3-D [batch_size, height, width]
    max_scale: float
        zoom-in scale of the image, make sure scale > 1 (means zoom in).
    seed_scale: int or None
    seed_shift: int or None
    name: str
        operation name

    Returns
    -------
    A zoom-in tensor of the same shape as `image`

    """
    with tf.name_scope(name, "random_zoom_in", [image, max_scale]):
        image_shape = image.get_shape()
        if image_shape.ndims == 3 or image_shape.ndims is None:
            rd_scale = tf.random_uniform([2], 1, max_scale, seed=seed_scale)
            shape = tf.shape(image)
            size = tf.to_int32(tf.to_float(shape[:-1]) * rd_scale)
            expanded_image = tf.expand_dims(image, axis=0)
            central_zoom_in_image = tf.image.resize_bilinear(expanded_image, size)
            central_zoom_in_image = tf.squeeze(central_zoom_in_image, axis=0)
            seed = random.randint(0, 2147482647) if seed_shift is None else seed_shift

            cropped_label = None
            if label is not None:
                label_shape = label.get_shape()
                assert label_shape.ndims is None or image_shape.ndims is None or \
                    image_shape.ndims - label_shape.ndims == 1, \
                    "image and label shape: {} vs {}".format(image_shape, label_shape)
                # Zoom in also with label
                expanded_label = tf.expand_dims(tf.expand_dims(label, 0), -1)
                central_zoom_in_label = tf.image.resize_nearest_neighbor(expanded_label, size)
                central_zoom_in_label = tf.to_float(tf.squeeze(central_zoom_in_label, axis=0))

                combined = tf.concat(axis=-1, values=[central_zoom_in_image, central_zoom_in_label])
                new_shape = tf.concat([shape[:-1], [shape[-1] + 1]], axis=0)
                cropped_combined = tf.image.random_crop(combined, new_shape, seed=seed)
                cropped_image = cropped_combined[..., :-1]
                cropped_label = tf.to_int32(cropped_combined[..., -1])
            else:
                cropped_image = tf.image.random_crop(central_zoom_in_image, shape, seed=seed)

        elif image_shape.ndims == 4:
            rd_scale = tf.random_uniform([2], 1, max_scale, seed=seed_scale)
            shape = tf.shape(image)
            size = tf.to_int32(tf.to_float(shape[1:-1]) * rd_scale)
            central_zoom_in_image = tf.image.resize_bilinear(image, size)
            seed = random.randint(0, 2147482647) if seed_shift is None else seed_shift

            cropped_label = None
            if label is not None:
                label_shape = label.get_shape()
                assert label_shape.ndims is None or image_shape.ndims is None or \
                    image_shape.ndims - label_shape.ndims == 1, \
                    "image and label shape: {} vs {}".format(image_shape, label_shape)
                # Zoom in also with label
                extended_label = tf.expand_dims(label, axis=-1)
                central_zoom_in_label = tf.image.resize_nearest_neighbor(extended_label, size)
                central_zoom_in_label = tf.to_float(central_zoom_in_label)

                combined = tf.concat(axis=-1, values=[central_zoom_in_image, central_zoom_in_label])
                new_shape = tf.concat([shape[:-1], [shape[-1] + 1]], axis=0)
                cropped_combined = tf.image.random_crop(combined, new_shape, seed=seed)
                cropped_image = cropped_combined[..., :-1]
                cropped_label = tf.to_int32(cropped_combined[..., -1])
            else:
                cropped_image = tf.image.random_crop(central_zoom_in_image, shape, seed=seed)
        else:
            raise ValueError('\'image\' must have either 3 or 4 dimensions.')

        return cropped_image, cropped_label


def random_noise(image, scale, seed=None, name=None):
    """
    Add a random noise tensor to image.

    Parameters
    ----------
    image: Tensor
        N-D image
    scale: Scalar
        noise scale.
        Notice that we use random_uniform to generate noise tensor.
    seed: Scalar
        random seed
    name: str
        operation name

    Returns
    -------
    A new tensor of the same shape as `image` with some noise

    """
    with tf.name_scope(name, "random_noise", [image, scale]):
        shape = tf.shape(image)
        abs_scale = tf.abs(scale)
        rd_tensor = tf.random_uniform(shape, -abs_scale, abs_scale, seed=seed)
        noised = tf.add(image, rd_tensor, name="NoisedImage")

        return noised


def random_flip_left_right(image, label=None, seed=None):
    """Randomly flip an image horizontally (left to right).

    With a 1 in 2 chance, outputs the contents of `image` flipped along the
    second dimension, which is `width`.  Otherwise output the image as-is.

    Parameters
    ----------
    image: 4-D Tensor of shape `[batch, height, width, channels]` or
           3-D Tensor of shape `[height, width, channels]`.
    label: 3-D Tensor of shape `[batch, height, width]` or
           2-D Tensor of shape `[height, width]`.
    seed: A Python integer. Used to create a random seed. See
        `tf.set_random_seed`
        for behavior.

    Returns
    -------
    A tensor of the same type and shape as `image`.

    Raises
    ------
    ValueError: if the shape of `image` not supported.
    """
    if label is None:
        return tf.image.random_flip_left_right(image, seed)
    else:
        label_new = tf.expand_dims(tf.cast(label, image.dtype), axis=-1)
        combined = tf.concat((image, label_new), axis=-1)
        combined_flipped = tf.image.random_flip_left_right(combined, seed)
        return combined_flipped[..., :-1], tf.cast(combined_flipped[..., -1], dtype=label.dtype)
