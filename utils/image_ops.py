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


def zscore(img):
    nonzero_region = img > 0
    flatten_img = tf.reshape(img, [-1])
    flatten_mask = tf.reshape(nonzero_region, [-1])
    mean, variance = tf.nn.moments(tf.boolean_mask(flatten_img, flatten_mask), axes=(0,))
    float_region = tf.cast(nonzero_region, img.dtype)
    new_img = (img - float_region * mean) / (float_region * tf.math.sqrt(variance) + 1e-8)
    return new_img


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
    with tf.name_scope(name, 'random_adjust_wwl', [image, w_width, w_level, seed1, seed2]):
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
            size = tf.cast(tf.cast(shape[:-1], tf.float32) * rd_scale, tf.int32)
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
                central_zoom_in_label = tf.cast(tf.squeeze(central_zoom_in_label, axis=0), tf.float32)

                combined = tf.concat(axis=-1, values=[central_zoom_in_image, central_zoom_in_label])
                new_shape = tf.concat([shape[:-1], [shape[-1] + 1]], axis=0)
                cropped_combined = tf.image.random_crop(combined, new_shape, seed=seed)
                cropped_image = cropped_combined[..., :-1]
                cropped_label = tf.cast(cropped_combined[..., -1], tf.int32)
            else:
                cropped_image = tf.image.random_crop(central_zoom_in_image, shape, seed=seed)

        elif image_shape.ndims == 4:
            rd_scale = tf.random_uniform([2], 1, max_scale, seed=seed_scale)
            shape = tf.shape(image)
            size = tf.cast(tf.cast(shape[1:-1], tf.float32) * rd_scale, tf.int32)
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
                central_zoom_in_label = tf.cast(central_zoom_in_label, tf.float32)

                combined = tf.concat(axis=-1, values=[central_zoom_in_image, central_zoom_in_label])
                new_shape = tf.concat([shape[:-1], [shape[-1] + 1]], axis=0)
                cropped_combined = tf.image.random_crop(combined, new_shape, seed=seed)
                cropped_image = cropped_combined[..., :-1]
                cropped_label = tf.cast(cropped_combined[..., -1], tf.int32)
            else:
                cropped_image = tf.image.random_crop(central_zoom_in_image, shape, seed=seed)
        else:
            raise ValueError('\'image\' must have either 3 or 4 dimensions.')

        return cropped_image, cropped_label


def random_crop(image, label, shape, seed=None, name=None):
    with tf.name_scope(name, "random_crop", [image, shape]):
        image_shape = image.get_shape()
        if image_shape.ndims == 3 or image_shape.ndims is None:
            label_shape = label.get_shape()
            assert label_shape.ndims is None or image_shape.ndims is None or \
                image_shape.ndims - label_shape.ndims == 1, \
                "image and label shape: {} vs {}".format(image_shape, label_shape)
            # Zoom in also with label
            expanded_label = tf.cast(tf.expand_dims(label, -1), tf.float32)
            combined = tf.concat(axis=-1, values=[image, expanded_label])
            shape = list(shape[:-1]) + [shape[-1] + 1]
            cropped_combined = tf.image.random_crop(combined, shape, seed=seed)
            cropped_image = cropped_combined[..., :-1]
            cropped_label = tf.cast(cropped_combined[..., -1], tf.int32)
        else:
            raise ValueError('\'image\' must have either 3 or 4 dimensions.')

        return cropped_image, cropped_label


def random_noise(image, scale, mask=None, seed=None, name=None, ntype="uniform"):
    """
    Add a random noise tensor to image.

    Parameters
    ----------
    image: Tensor
        N-D image
    scale: Scalar
        noise scale.
        Notice that we use random_uniform to generate noise tensor.
    mask: Tensor
        noise mask
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
        if ntype == "uniform":
            rd_tensor = tf.random_uniform(shape, -abs_scale, abs_scale, seed=seed, dtype=image.dtype)
        else:
            rd_tensor = tf.random_normal(shape, stddev=abs_scale, seed=seed, dtype=image.dtype)

        if mask is None:
            new_image = tf.add(image, rd_tensor, name="NoisedImage")
        else:
            new_image = tf.add(image, rd_tensor * mask, name="NoisedImage")

        return new_image


def random_flip(image, label=None, seed=None, direction="lr", name=None):
    """Randomly flip an image horizontally (left to right) or vertically (up to down).

    With a 1 in 2 chance, outputs the contents of `image` flipped along the
    specified dimension.  Otherwise output the image as-is.

    Parameters
    ----------
    image: 4-D Tensor of shape `[batch, height, width, channels]` or
           3-D Tensor of shape `[height, width, channels]`.
    label: 3-D Tensor of shape `[batch, height, width]` or
           2-D Tensor of shape `[height, width]`.
    seed: A Python integer. Used to create a random seed. See
        `tf.set_random_seed`
        for behavior.
    direction: str
        `lr` for left/right, `ud` for up/down

    Returns
    -------
    A tensor of the same type and shape as `image`.

    Raises
    ------
    ValueError: if the shape of `image` not supported.
    """
    with tf.name_scope(name, "random_flip", [image, label]):
        if label is None:
            if direction == "lr":
                return tf.image.random_flip_left_right(image, seed)
            elif direction == "ud":
                return tf.image.random_flip_up_down(image, seed)
            else:
                raise ValueError("Wrong direction: %s" % direction)
        else:
            label_new = tf.expand_dims(tf.cast(label, image.dtype), axis=-1)
            combined = tf.concat((image, label_new), axis=-1)
            if direction == "lr":
                combined_flipped = tf.image.random_flip_left_right(combined, seed)
            elif direction == "ud":
                combined_flipped = tf.image.random_flip_up_down(combined, seed)
            else:
                raise ValueError("Wrong direction: %s" % direction)
            return combined_flipped[..., :-1], tf.cast(combined_flipped[..., -1], dtype=label.dtype)


def random_flip_left_right(image, label=None, seed=None):
    return random_flip(image, label, seed, direction="lr", name="random_flip_left_right")


def random_flip_up_down(image, label=None, seed=None):
    return random_flip(image, label, seed, direction="ud", name="random_flip_up_down")


def random_zero_or_one(shape, dtype, seed=None):
    """
    Randomly return a zero/one tensor.

    Parameters
    ----------
    shape: TensorShape
    dtype:
    seed: A Python integer. Used to create a random seed. See
        `tf.set_random_seed`
        for behavior.

    Returns
    -------
    A tensor with shape as `shape`
    """
    rd_val = tf.random_uniform((), 0, 1, dtype=dtype, seed=seed)
    rd_rnd = tf.round(rd_val)
    return tf.fill(shape, rd_rnd)


def augment_gamma(image, gamma_range, retain_stats=False, p_per_sample=1, epsilon=1e-7):
    if retain_stats:
        mn, variance = tf.nn.moments(image, axes=(0, 1, 2))
        sd = tf.math.sqrt(variance)
    gamma = tf.cond(tf.random_uniform((), 0, 1, dtype=tf.float32) < p_per_sample,
                    lambda: tf.random_uniform((), gamma_range[0], 1),
                    lambda: tf.random_uniform((), 1, gamma_range[1]))
    minm = tf.reduce_min(image)
    rnge = tf.reduce_max(image) - minm
    new_image = tf.math.pow((image - minm) / (rnge + epsilon), gamma) * rnge + minm
    if retain_stats:
        new_mn, new_variance = tf.nn.moments(new_image, axes=(0, 1, 2))
        new_image = new_image - new_mn + mn
        new_image = new_image / (tf.math.sqrt(new_variance) + 1e-8) * sd
    return new_image


def binary_dilation2d(inputs, connection=1, iterations=1, padding="SAME", name=None):
    """
    Computes the gray-scale binary dilation of 4-D input

    Parameters
    ----------
    inputs: Tensor.
        Must be one of the following types: float32, float64, int32, uint8, int16, int8,
        int64, bfloat16, uint16, half, uint32, uint64. 4-D with shape [batch, in_height, in_width, depth].
    connection: int
        If connection == 1, then [[0, 1, 0], [1, 1, 1], [0, 1, 0]] will be the filter.
        If connection == 2, then [[1, 1, 1], [1, 1, 1], [1, 1, 1]] will be the filter.
    iterations: int
        Iteration numbers
    padding: str
        From: "SAME", "VALID". The type of padding algorithm to use.
    name: str
        A name for the operation (optional).

    Returns
    -------
    A Tensor. Has the same type as input.
    """
    if connection == 1:
        kernel_array = [0, 1, 0, 1, 1, 1, 0, 1, 0]
    elif connection == 2:
        kernel_array = [1, 1, 1, 1, 1, 1, 1, 1, 1]
    else:
        raise ValueError("connection must be 1 or 2, got {}".format(connection))
    kernel = tf.constant(kernel_array, dtype=tf.int32, shape=(3, 3, 1), name="DilationFilter")
    kernel = tf.tile(kernel, tf.concat(([1, 1], [tf.shape(inputs)[-1]]), axis=0))

    outputs = inputs
    for _ in range(iterations):
        outputs = tf.nn.dilation2d(outputs, kernel, [1, 1, 1, 1],
                                   [1, 1, 1, 1], padding, name) - 1
    return outputs


def create_spatial_guide_2d(shape, center, stddev):
    """
    Tensorflow implementation of `create_gaussian_distribution()`

    Parameters
    ----------
    shape: Tensor
        two values
    center: Tensor
        Float tensor with shape [n, 2], 2 means (x, y)
    stddev: Tensor
        Float tensor with shape [n, 2], 2 means (x, y)

    Returns
    -------
    A batch of spatial guide image

    Warnings
    --------
    User must make sure stddev doesn't contain zero.

    Notes
    -----
    -1s in center and stddev are padding value and almost don't affect spatial guide
    """
    y = tf.range(shape[0])
    x = tf.range(shape[1])
    # Let n the number of tumors in current slice
    coords = tf.tile(tf.expand_dims(
        tf.stack(tf.meshgrid(y, x, indexing="ij"), axis=-1), axis=0),
        multiples=tf.concat((tf.shape(center)[:1], [1, 1, 1]), axis=0))     # [n, h, w, 2]
    coords = tf.cast(coords, tf.float32)
    center = tf.expand_dims(tf.expand_dims(center, axis=-2), axis=-2)       # [n, 1, 1, 2]
    stddev = tf.expand_dims(tf.expand_dims(stddev, axis=-2), axis=-2)       # [n, 1, 1, 2]
    normalizer = 2. * stddev * stddev                                       # [n, 1, 1, 2]
    d = tf.exp(-tf.reduce_sum((coords - center) ** 2 / normalizer, axis=-1, keepdims=True))    # [n, h, w]
    return tf.reduce_max(d, axis=0)               # [h, w, 1]
