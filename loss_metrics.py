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
from medpy import metric as mtr     # pip install medpy

METRICS = "metrics"


def get_losses(graph=None):
    graph = graph or tf.get_default_graph()
    total_losses = tf.losses.get_losses() + [get_or_create_regularization_loss(graph)]
    for loss in total_losses:
        yield loss


def get_or_create_regularization_loss(graph, regu_loss_name="Losses/total_regularization_loss:0"):
    try:
        regu_loss = graph.get_tensor_by_name(regu_loss_name)
    except KeyError:
        with tf.name_scope("Losses/"):
            regu_loss = tf.losses.get_regularization_loss()
    return regu_loss


def get_total_loss(add_regularization_losses=True, graph=None, name="total_loss"):
    """Returns a tensor whose value represents the total loss.

      In particular, this adds any losses you have added with `tf.add_loss()` to
      any regularization losses that have been added by regularization parameters
      on layers constructors e.g. `tf.layers`. Be very sure to use this if you
      are constructing a loss_op manually. Otherwise regularization arguments
      on `tf.layers` methods will not function.

      Args:
        add_regularization_losses: A boolean indicating whether or not to use the
          regularization losses in the sum.
        graph: A Graph object
        name: The name of the returned tensor.

      Returns:
        A `Tensor` whose value represents the total loss.

      Raises:
        ValueError: if `losses` is not iterable.
      """
    losses = tf.losses.get_losses()
    graph = graph or tf.get_default_graph()

    if add_regularization_losses:
        losses += [get_or_create_regularization_loss(graph)]
    return tf.add_n(losses, name=name)


def _compute_weights(w_type, one_hot_labels, name=None, **kwargs):
    w_type = w_type.lower()
    # num_cls = one_hot_labels.shape[-1]
    bs = one_hot_labels.shape[0]
    ndim = len(one_hot_labels.shape)

    with tf.name_scope(name, "_compute_weights"):
        if w_type == "none":
            w = 1.0
        elif w_type == "numerical":
            if "numeric_w" not in kwargs:
                raise KeyError("w_type `numerical` need keyword argument `numeric_w`")
            numeric_w = kwargs["numeric_w"]
            w = tf.constant([numeric_w for _ in range(bs)], dtype=tf.float32)
        elif w_type == "proportion":
            num_labels = tf.reduce_sum(one_hot_labels, axis=range(1, ndim - 1))
            if "proportion_decay" in kwargs:
                num_labels += kwargs["proportion_decay"]
            proportions = 1.0 / num_labels
            w = proportions / tf.reduce_sum(proportions)
        else:
            raise ValueError("Not supported weight type: " + w_type)

        return w


def sparse_softmax_cross_entropy(logits, labels):
    return tf.losses.sparse_softmax_cross_entropy(labels, logits, scope="SceLoss")


def weighted_sparse_softmax_cross_entropy(logits, labels, w_type, name=None, **kwargs):
    with tf.name_scope("LabelProcess"):
        num_classes = logits.shape[-1]
        one_hot_labels = tf.one_hot(labels, num_classes)
    with tf.name_scope(name, "WsceLoss"):
        weights = _compute_weights(w_type, one_hot_labels, **kwargs)
        return tf.losses.sparse_softmax_cross_entropy(labels, logits, weights)


################################################################################################
#
#   API for metrics
#
################################################################################################
def get_metrics():
    for metric in tf.get_collection(METRICS):
        yield metric


def metric_dice(logits, labels, eps=1e-5, collections=METRICS, name=None):
    """
    Dice coefficient for N-D Tensor.

    Support "soft dice", which means logits and labels don't need to be binary tensors.

    Parameters
    ----------
    logits: tf.Tensor
        shape [batch_size, ..., 1]
    labels: tf.Tensor
        shape [batch_size, ..., 1]
    eps: float
        epsilon is set to avoid dividing zero
    collections: str
        collections to collect metrics
    name: str
        operation name used in tensorflow

    Returns
    -------
    Average dice coefficient
    """
    dim = len(logits.get_shape())
    sum_axis = list(range(1, dim))
    with tf.name_scope(name, "Dice", [logits, labels, eps]):
        logits = tf.cast(logits, tf.float32)
        labels = tf.cast(labels, tf.float32)

        intersection = tf.reduce_sum(logits * labels, axis=sum_axis)
        left = tf.reduce_sum(logits, axis=sum_axis)
        right = tf.reduce_sum(labels, axis=sum_axis)
        dice = (2 * intersection) / (left + right + eps)
        dice = tf.reduce_mean(dice, name="value")
        tf.add_to_collection(collections, dice)
        return dice


def metric_voe(logits, labels, eps=1e-5, collections=METRICS, name=None):
    """ Volumetric Overlap Error for N-D Tensor """
    dim = len(logits.get_shape())
    sum_axis = list(range(1, dim))
    with tf.name_scope(name, "VOE", [logits, labels, eps]):
        logits = tf.cast(logits, tf.float32)
        labels = tf.cast(labels, tf.float32)

        numerator = tf.reduce_sum(logits * labels, axis=sum_axis)
        denominator = tf.reduce_sum(tf.clip_by_value(logits + labels, 0.0, 1.0), axis=sum_axis)
        voe = 100 * (1.0 - numerator / (denominator + eps))
        voe = tf.reduce_mean(voe, name="value")
        tf.add_to_collection(collections, voe)
        return voe


def metric_vd(logits, labels, eps=1e-5, collections=METRICS, name=None):
    """ Relative Volume Difference for N-D Tensor """
    dim = len(logits.get_shape())
    sum_axis = list(range(1, dim))
    with tf.name_scope(name, "VD", [logits, labels, eps]):
        logits = tf.cast(logits, tf.float32)
        labels = tf.cast(labels, tf.float32)

        a = tf.reduce_sum(logits, axis=sum_axis)
        b = tf.reduce_sum(labels, axis=sum_axis)
        vd = 100 * (tf.abs(a - b) / (b + eps))
        vd = tf.reduce_mean(vd, name="value")
        tf.add_to_collection(collections, vd)
        return vd


def metric_3d(logits3d, labels3d, required=None, **kwargs):
    """
    Compute 3D metrics:

    * (Dice) Dice Coefficient

    * (VOE)  Volumetric Overlap Error

    * (VD)   Relative Volume Difference

    * (ASD)  Average Symmetric Surface Distance

    * (RMSD) Root Mean Square Symmetric Surface Distance

    * (MSD)  Maximum Symmetric Surface Distance

    Parameters
    ----------
    logits3d: ndarray
        3D binary prediction, shape is the same with `labels3d`, it should be an int
        array or boolean array.
    labels3d: ndarray
        3D labels for segmentation, shape [None, None, None], it should be an int array
        or boolean array. If the dimensions of `logits3d` and `labels3d` are greater than
        3, then `np.squeeze` will be applied to remove extra single dimension and then
        please make sure these two variables are still have 3 dimensions. For example,
        shape [None, None, None, 1] or [1, None, None, None, 1] are allowed.
    required: str or list
        a string or a list of string to specify which metrics need to be return, default
        this function will return all the metrics listed above. For example, if use
        ```python
        _metric_3D(logits3D, labels3D, require=["Dice", "VOE", "ASD"])
        ```
        then only these three metrics will be returned.
    kwargs: dict
        sampling: list
            the pixel resolution or pixel size. This is entered as an n-vector where n
            is equal to the number of dimensions in the segmentation i.e. 2D or 3D. The
            default value is 1 which means pixls are 1x1x1 mm in size

    Returns
    -------
    metrics required

    Notes
    -----
    Thanks to the code snippet from @MLNotebook's blog.

    [Blog link](https://mlnotebook.github.io/post/surface-distance-function/).
    """
    metrics = ["Dice", "VOE", "RVD", "ASSD", "RMSD", "MSD"]
    need_dist_map = False

    if required is None:
        required = metrics
    elif isinstance(required, str):
        required = [required]
        if required[0] not in metrics:
            raise ValueError("Not supported metric: %s" % required[0])
        elif required in metrics[3:]:
            need_dist_map = True
        else:
            need_dist_map = False

    for req in required:
        if req not in metrics:
            raise ValueError("Not supported metric: %s" % req)
        if (not need_dist_map) and req in metrics[3:]:
            need_dist_map = True

    if logits3d.ndim > 3:
        logits3d = np.squeeze(logits3d)
    if labels3d.ndim > 3:
        labels3d = np.squeeze(labels3d)

    assert logits3d.shape == labels3d.shape, ("Shape mismatch of logits3D and labels3D. \n"
                                              "Logits3D has shape %r while labels3D has "
                                              "shape %r" % (logits3d.shape, labels3d.shape))
    logits3d = logits3d.astype(np.bool)
    labels3d = labels3d.astype(np.bool)

    metrics_3d = {}
    sampling = kwargs.get("sampling", [1., 1., 1.])

    if need_dist_map:
        from utils.surface import Surface
        if np.count_nonzero(logits3d) == 0 or np.count_nonzero(labels3d) == 0:
            metrics_3d['ASSD'] = 0
            metrics_3d['MSD'] = 0
        else:
            eval_surf = Surface(logits3d, labels3d, physical_voxel_spacing=sampling,
                                mask_offset=[0., 0., 0.],
                                reference_offset=[0., 0., 0.])

            if "ASSD" in required:
                metrics_3d["ASSD"] = eval_surf.get_average_symmetric_surface_distance()
                required.remove("ASSD")
            if "MSD" in required:
                metrics_3d["MSD"] = eval_surf.get_maximum_symmetric_surface_distance()
            if "RMSD" in required:
                metrics_3d["RMSD"] = eval_surf.get_root_mean_square_symmetric_surface_distance()

    if required:
        if "Dice" in required:
            metrics_3d["Dice"] = mtr.dc(logits3d, labels3d)
        if "VOE" in required:
            metrics_3d["VOE"] = 1. - mtr.jc(logits3d, labels3d)
        if "RVD" in required:
            metrics_3d["RVD"] = mtr.ravd(logits3d, labels3d)

    return metrics_3d
