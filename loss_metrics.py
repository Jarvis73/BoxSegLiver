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
from utils import array_kits
from medpy import metric as mtr     # pip install medpy
from scipy.ndimage.morphology import distance_transform_edt

METRICS = "metrics"


def add_arguments(parser):
    group = parser.add_argument_group(title="Loss Arguments")
    group.add_argument("--weight_decay_rate",
                       type=float,
                       default=1e-5,
                       required=False, help="Weight decay rate for variable regularizers (default: %(default)f)")
    group.add_argument("--bias_decay",
                       action="store_true",
                       required=False, help="Use bias decay or not")
    group.add_argument("--loss_type",
                       type=str,
                       default="xentropy",
                       choices=["xentropy", "dice", "xentropy+dice"],
                       required=False, help="Loss type (default %(default)s)")
    group.add_argument("--loss_weight_type",
                       type=str,
                       default="none",
                       choices=["none", "numerical", "proportion", "boundary"],
                       required=False, help="Weights used in loss function for alleviating class imbalance problem "
                                            "(default %(default)s)")
    group.add_argument("--loss_numeric_w",
                       type=float,
                       nargs="+",
                       required=False, help="Numeric weights for loss_weight_type=\"numerical\". Notice that one value"
                                            "for one class")
    group.add_argument("--loss_proportion_decay",
                       type=float,
                       default=1000,
                       required=False, help="Proportion decay for loss_weight_type=\"proportion\". Check source code"
                                            "for details. (default: %(default)f)")
    group.add_argument("--metrics_train",
                       type=str,
                       default=["Dice"],
                       choices=["Dice", "VOE", "VD"],
                       nargs="+",
                       required=False, help="Evaluation metric names (default: %(default)s)")
    group.add_argument("--metrics_eval",
                       type=str,
                       default=["Dice"],
                       choices=["Dice", "VOE", "RVD", "ASSD", "RMSD", "MSD"],
                       nargs="+",
                       required=False, help="Evaluation metric names (default: %(default)s)")


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
    size = tf.cast(tf.shape(one_hot_labels), tf.float32)
    ndim = len(one_hot_labels.shape)

    with tf.name_scope(name, "_compute_weights"):
        if w_type == "none":
            return tf.constant(1.0, dtype=tf.float32)
        elif w_type == "numerical":
            if "numeric_w" not in kwargs:
                raise KeyError("w_type `numerical` need keyword argument `numeric_w`")
            numeric_w = kwargs["numeric_w"]
            w = tf.constant([numeric_w for _ in range(bs)], dtype=tf.float32)
            if ndim == 4:
                w = tf.reduce_sum(w[:, None, None, :] * one_hot_labels, axis=-1)  # [bs, h, w]
            else:   # ndim == 5
                w = tf.reduce_sum(w[:, None, None, None, :] * one_hot_labels, axis=-1)  # [bs, d, h, w]
        elif w_type == "proportion":
            num_labels = tf.reduce_sum(one_hot_labels, axis=list(range(1, ndim - 1)))
            if "proportion_decay" in kwargs:
                num_labels += kwargs["proportion_decay"]
            proportions = 1.0 / num_labels
            w = proportions / tf.reduce_sum(proportions, axis=1, keepdims=True)
            if ndim == 4:
                w = tf.reduce_sum(w[:, None, None, :] * one_hot_labels, axis=-1)  # [bs, h, w]
            else:   # ndim == 5
                w = tf.reduce_sum(w[:, None, None, None, :] * one_hot_labels, axis=-1)  # [bs, d, h, w]
        elif w_type == "examples":
            if ndim == 4:
                w = kwargs["examples_w"][:, None, None]
            else:   # ndim == 5
                w = kwargs["examples_w"][:, None, None, None]
        elif w_type == "boundary":
            shp = one_hot_labels.shape
            reshaped = tf.reshape(tf.transpose(one_hot_labels, (0, 3, 1, 2)), [shp[0] * shp[3], shp[1], shp[2], 1])
            kernel = tf.ones((3, 3, 1, 1), dtype=tf.float32)
            dilated = tf.clip_by_value(tf.nn.conv2d(reshaped, kernel, [1] * 4, "SAME"), 0, 1) - reshaped
            seperated = tf.reshape(dilated, [shp[0], shp[3], shp[1], shp[2]])
            weight = tf.logical_not(tf.cast(tf.reduce_sum(seperated, axis=1), tf.bool))
            weight = tf.map_fn(lambda x: tf.py_func(
                lambda y: distance_transform_edt(y).astype(np.float32), [x], tf.float32), weight, tf.float32)
            w = tf.exp(-weight / 25) + 1
            tf.summary.image(f"{kwargs['tag']}/Weight", tf.expand_dims(w, axis=-1), max_outputs=1)
        else:
            raise ValueError("Not supported weight type: " + w_type)
        # norm
        w = w / tf.reduce_sum(w, axis=list(range(1, ndim - 1)), keepdims=True) * \
            (size[1] * size[2] if ndim == 4 else size[1] * size[2] * size[3])
        return w


def sparse_softmax_cross_entropy(logits, labels):
    return tf.losses.sparse_softmax_cross_entropy(labels, logits, scope="SceLoss")


def weighted_sparse_softmax_cross_entropy(logits, labels, w_type, name=None, **kwargs):
    with tf.name_scope(name, "WsceLoss"):   # weighted softmax cross entropy --> Wsce
        num_classes = logits.shape[-1]
        one_hot_labels = tf.one_hot(labels, num_classes)
        weights = _compute_weights(w_type, one_hot_labels, **kwargs)
        return tf.losses.sparse_softmax_cross_entropy(labels, logits, weights)


def sparse_dice_loss(_sentinel=None, labels=None, logits=None,
                     with_bg=False, eps=1e-8, name=None,
                     loss_collection=tf.GraphKeys.LOSSES):
    """ Implementation of multi-class dice loss(without background), also
    called Generalized Dice Loss(GDL).

    Parameters
    ----------
    _sentinel: None
        not used
    labels: Tensor
        sparse labels with shape [bs, h, w]
    logits: Tensor
        probability of each pixel after softmax layer
    with_bg: bool
        with background or not
    eps: float
        used to avoid dividing by zero
    name: str
        layer name used in computation graph
    loss_collection: str
        collection to which the loss will be added.

    Returns
    -------
    Dice loss, between [0, 1]
    """
    with tf.variable_scope(name, "DiceLoss"):
        n_classes = logits.shape.as_list()[-1]
        c_logits = tf.cast(logits, tf.float32)
        h_labels = tf.one_hot(labels, n_classes, dtype=tf.float32)

        if not with_bg:
            n_classes -= 1
            c_logits = c_logits[..., 1:]
            h_labels = h_labels[..., 1:]

        tf_sum = tf.reduce_sum
        intersection = tf_sum(h_labels * c_logits, [1, 2, 3])
        union = tf_sum(h_labels + c_logits, [1, 2, 3])
        mean_dice_loss = tf.reduce_mean((2.0 * intersection) / (union + eps))
        del tf_sum

        dice_loss = 1. - mean_dice_loss

    tf.losses.add_loss(dice_loss, loss_collection)
    return dice_loss


def weighted_dice_loss(logits, labels, w_type, name=None, **kwargs):
    with tf.name_scope(name, "WsdLoss"):    # weighted softmax dice --> Wsd
        return sparse_dice_loss(labels=labels, logits=logits)


def sparse_focal_loss(prediction_tensor, target_tensor, alpha=0.25, gamma=2):
    with tf.name_scope("FocalLoss"):
        target_tensor = tf.reshape(target_tensor, [-1, 1])
        one_minus_target = 1 - target_tensor
        merged_target = tf.cast(tf.concat((one_minus_target, target_tensor), axis=1), tf.float32)
        sigmoid_p = tf.nn.sigmoid(prediction_tensor)
        zeros = tf.zeros_like(sigmoid_p, dtype=sigmoid_p.dtype)

        pos_p_sub = tf.where(merged_target > zeros, merged_target - sigmoid_p, zeros)

        neg_p_sub = tf.where(merged_target > zeros, zeros, sigmoid_p)
        per_entry_cross_ent = (pos_p_sub ** gamma) * tf.nn.softplus(-prediction_tensor) + \
                              (neg_p_sub ** gamma) * tf.nn.softplus(prediction_tensor)

        return tf.reduce_mean(tf.reduce_sum(per_entry_cross_ent, axis=1))


################################################################################################
#
#   API for metrics
#
################################################################################################
def get_metrics():
    for metric in tf.get_collection(METRICS):
        yield metric


def metric_dice(logits, labels, eps=1e-5, collections=METRICS, name=None, reduce=True):
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
    reduce: bool
        reduce metrics by mean or not

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
        dice = (2 * intersection + eps) / (left + right + eps)
        if reduce:
            dice = tf.reduce_mean(dice, name="value")
        else:
            dice = tf.identity(dice, name="value")
        tf.add_to_collection(collections, dice)
        return dice


def metric_voe(logits, labels, eps=1e-5, collections=METRICS, name=None, reduce=True):
    """ Volumetric Overlap Error for N-D Tensor """
    dim = len(logits.get_shape())
    sum_axis = list(range(1, dim))
    with tf.name_scope(name, "VOE", [logits, labels, eps]):
        logits = tf.cast(logits, tf.float32)
        labels = tf.cast(labels, tf.float32)

        numerator = tf.reduce_sum(logits * labels, axis=sum_axis)
        denominator = tf.reduce_sum(tf.clip_by_value(logits + labels, 0.0, 1.0), axis=sum_axis)
        voe = 100 * (1.0 - numerator / (denominator + eps))
        if reduce:
            voe = tf.reduce_mean(voe, name="value")
        else:
            voe = tf.identity(voe, name="value")
        tf.add_to_collection(collections, voe)
        return voe


def metric_vd(logits, labels, eps=1e-5, collections=METRICS, name=None, reduce=True):
    """ Relative Volume Difference for N-D Tensor """
    dim = len(logits.get_shape())
    sum_axis = list(range(1, dim))
    with tf.name_scope(name, "VD", [logits, labels, eps]):
        logits = tf.cast(logits, tf.float32)
        labels = tf.cast(labels, tf.float32)

        a = tf.reduce_sum(logits, axis=sum_axis)
        b = tf.reduce_sum(labels, axis=sum_axis)
        vd = 100 * (tf.abs(a - b) / (b + eps))
        if reduce:
            vd = tf.reduce_mean(vd, name="value")
        else:
            vd = tf.identity(vd, name="value")
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
            metrics_3d["RVD"] = np.absolute(mtr.ravd(logits3d, labels3d))

    return metrics_3d


def tumor_detection_metrics(result, reference, iou_thresh=0.5, connectivity=1,
                            verbose=False, logger=None, name=""):
    """ Compute true positive, false positive, true negative

    Parameters
    ----------
    result: ndarray
        3D, range in {0, 1}
    reference:  ndarray
        3D, range in {0, 1}
    iou_thresh: float
        threshold for determining if two objects are correlated
    connectivity: int
        passes to `generate_binary_structure`
    verbose: show output or not
    logger: Logger
    name: str
        name of case
    """
    _, _, n_res, n_ref, mapping = array_kits.distinct_binary_object_correspondences(
        result, reference, iou_thresh, connectivity
    )

    tp = len(mapping)
    fp = n_res - tp
    if n_res != 0:
        precision = tp / n_res
    else:
        precision = np.inf
    if n_ref != 0:
        recall = tp / n_ref
    else:
        recall = np.inf

    ret = {"tp": tp,
           "fp": fp,
           "pos": n_ref,
           "precision": precision,
           "recall": recall}

    if verbose:
        info = ("{:s} TPs: {:3d} FPs: {:3d} Pos: {:3d} Precision: {:.3f} Recall: {:.3f}"
                .format(name, tp, fp, n_ref, tp / (tp + fp), recall))
        if logger is not None:
            logger.info(info)
        else:
            print(info)

    return ret


class ConfusionMatrix(object):
    def __init__(self, test=None, reference=None):
        self.tp = None
        self.fp = None
        self.tn = None
        self.fn = None
        self.size = None
        self.reference_empty = None
        self.reference_full = None
        self.test_empty = None
        self.test_full = None
        self.set_reference(reference)
        self.set_test(test)

    def set_test(self, test):

        self.test = test
        self.reset()

    def set_reference(self, reference):

        self.reference = reference
        self.reset()

    def reset(self):

        self.tp = None
        self.fp = None
        self.tn = None
        self.fn = None
        self.size = None
        self.test_empty = None
        self.test_full = None
        self.reference_empty = None
        self.reference_full = None

    def compute(self):
        if self.test is None or self.reference is None:
            raise ValueError("'test' and 'reference' must both be set to compute confusion matrix.")
        assert self.test.shape == self.reference.shape, "Shape mismatch: {} and {}".format(
            self.test.shape, self.reference.shape)

        self.tp = int(((self.test != 0) * (self.reference != 0)).sum())
        self.fp = int(((self.test != 0) * (self.reference == 0)).sum())
        self.tn = int(((self.test == 0) * (self.reference == 0)).sum())
        self.fn = int(((self.test == 0) * (self.reference != 0)).sum())
        self.size = self.reference.size
        self.test_empty = not np.any(self.test)
        self.test_full = np.all(self.test)
        self.reference_empty = not np.any(self.reference)
        self.reference_full = np.all(self.reference)

    def get_matrix(self):

        for entry in (self.tp, self.fp, self.tn, self.fn):
            if entry is None:
                self.compute()
                break

        return self.tp, self.fp, self.tn, self.fn

    def get_size(self):

        if self.size is None:
            self.compute()
        return self.size

    def get_existence(self):

        for case in (self.test_empty, self.test_full, self.reference_empty, self.reference_full):
            if case is None:
                self.compute()
                break

        return self.test_empty, self.test_full, self.reference_empty, self.reference_full
