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
import tensorflow_estimator as tfes
import tensorflow.contrib.slim as slim
from tensorflow.contrib.layers.python.layers import utils as utils_lib

import loss_metrics as losses
from NetworksV2 import base
from NetworksV2.Backbone import slim_nets
from utils import distribution_utils

ModeKeys = tfes.estimator.ModeKeys
metrics = losses


class _ModelConfig(object):
    config = {}

    config[4] = {
        "conv_e0": {"conv1": {"kernel": (1, 3, 3), "stride": (1, 1, 1)},
                    "conv2": {"kernel": (1, 3, 3), "stride": (1, 1, 1)}},
        "conv_e1": {"conv1": {"kernel": (1, 3, 3), "stride": (1, 2, 2)},
                    "conv2": {"kernel": (1, 3, 3), "stride": (1, 1, 1)}},
        "conv_e2": {"conv1": {"kernel": (3, 3, 3), "stride": (1, 2, 2)},
                    "conv2": {"kernel": (3, 3, 3), "stride": (1, 1, 1)}},
        "conv_e3": {"conv1": {"kernel": (3, 3, 3), "stride": (1, 2, 2)},
                    "conv2": {"kernel": (3, 3, 3), "stride": (1, 1, 1)}},

        "bridge":  {"conv1": {"kernel": (3, 3, 3), "stride": (2, 2, 2)},
                    "conv2": {"kernel": (3, 3, 3), "stride": (1, 1, 1)}},

        "conv_d3": {"up":    {"kernel": (2, 2, 2), "stride": (2, 2, 2)},
                    "conv1": {"kernel": (3, 3, 3), "stride": (1, 1, 1)},
                    "conv2": {"kernel": (3, 3, 3), "stride": (1, 1, 1)}},
        "conv_d2": {"up":    {"kernel": (1, 2, 2), "stride": (1, 2, 2)},
                    "conv1": {"kernel": (3, 3, 3), "stride": (1, 1, 1)},
                    "conv2": {"kernel": (3, 3, 3), "stride": (1, 1, 1)}},
        "conv_d1": {"up":    {"kernel": (1, 2, 2), "stride": (1, 2, 2)},
                    "conv1": {"kernel": (1, 3, 3), "stride": (1, 1, 1)},
                    "conv2": {"kernel": (1, 3, 3), "stride": (1, 1, 1)}},
        "conv_d0": {"up":    {"kernel": (1, 2, 2), "stride": (1, 2, 2)},
                    "conv1": {"kernel": (1, 3, 3), "stride": (1, 1, 1)},
                    "conv2": {"kernel": (1, 3, 3), "stride": (1, 1, 1)}},
    }

    config[5] = {
        "conv_e0": {"conv1": {"kernel": (1, 3, 3), "stride": (1, 1, 1)},
                    "conv2": {"kernel": (1, 3, 3), "stride": (1, 1, 1)}},
        "conv_e1": {"conv1": {"kernel": (1, 3, 3), "stride": (1, 2, 2)},
                    "conv2": {"kernel": (1, 3, 3), "stride": (1, 1, 1)}},
        "conv_e2": {"conv1": {"kernel": (3, 3, 3), "stride": (1, 2, 2)},
                    "conv2": {"kernel": (3, 3, 3), "stride": (1, 1, 1)}},
        "conv_e3": {"conv1": {"kernel": (3, 3, 3), "stride": (1, 2, 2)},
                    "conv2": {"kernel": (3, 3, 3), "stride": (1, 1, 1)}},
        "conv_e4": {"conv1": {"kernel": (3, 3, 3), "stride": (1, 2, 2)},
                    "conv2": {"kernel": (3, 3, 3), "stride": (1, 1, 1)}},

        "bridge":  {"conv1": {"kernel": (3, 3, 3), "stride": (2, 2, 2)},
                    "conv2": {"kernel": (3, 3, 3), "stride": (1, 1, 1)}},

        "conv_d4": {"up":    {"kernel": (2, 2, 2), "stride": (2, 2, 2)},
                    "conv1": {"kernel": (3, 3, 3), "stride": (1, 1, 1)},
                    "conv2": {"kernel": (3, 3, 3), "stride": (1, 1, 1)}},
        "conv_d3": {"up":    {"kernel": (1, 2, 2), "stride": (1, 2, 2)},
                    "conv1": {"kernel": (3, 3, 3), "stride": (1, 1, 1)},
                    "conv2": {"kernel": (3, 3, 3), "stride": (1, 1, 1)}},
        "conv_d2": {"up":    {"kernel": (1, 2, 2), "stride": (1, 2, 2)},
                    "conv1": {"kernel": (3, 3, 3), "stride": (1, 1, 1)},
                    "conv2": {"kernel": (3, 3, 3), "stride": (1, 1, 1)}},
        "conv_d1": {"up":    {"kernel": (1, 2, 2), "stride": (1, 2, 2)},
                    "conv1": {"kernel": (1, 3, 3), "stride": (1, 1, 1)},
                    "conv2": {"kernel": (1, 3, 3), "stride": (1, 1, 1)}},
        "conv_d0": {"up":    {"kernel": (1, 2, 2), "stride": (1, 2, 2)},
                    "conv1": {"kernel": (1, 3, 3), "stride": (1, 1, 1)},
                    "conv2": {"kernel": (1, 3, 3), "stride": (1, 1, 1)}},
    }


class UNet3D(base.BaseNet):
    def __init__(self, args, name=None):
        """ Don't create ops/tensors in __init__() """
        super(UNet3D, self).__init__(args)
        self.name = name or "UNet3D"
        self.classes.extend(self.args.classes)

        self.bs = distribution_utils.per_device_batch_size(args.batch_size, args.num_gpus)
        self.depth = args.im_depth
        self.height = args.im_height
        self.width = args.im_width
        self.channel = args.im_channel
        self.use_spatial = args.use_spatial

    def _net_arg_scope(self, *args, **kwargs):
        default_w_regu, default_b_regu = self._get_regularizer()
        default_w_init, _ = self._get_initializer()

        with slim.arg_scope([slim.conv3d, slim.conv3d_transpose],
                            weights_regularizer=default_w_regu,
                            weights_initializer=default_w_init,
                            biases_regularizer=default_b_regu):
            with slim.arg_scope([slim.max_pool3d], padding="SAME"):
                normalizer, params = self._get_normalization()
                with slim.arg_scope([slim.conv3d],
                                    normalizer_fn=normalizer,
                                    normalizer_params=params) as scope:
                    return scope

    def _build_network(self, *args, **kwargs):
        self.depth = base._check_size_type(self.depth)
        self.height = base._check_size_type(self.height)
        self.width = base._check_size_type(self.width)
        self._inputs["images"].set_shape([self.bs, self.depth, self.height, self.width, self.channel])
        if self.use_spatial:
            self._inputs["sp_guide"].set_shape([self.bs, self.depth, self.height, self.width, self.args.guide_channel])

        init_channels = kwargs.get("init_channels", 30)
        num_pool_layers = kwargs.get("num_pool_layers", 4)
        max_channels = kwargs.get("max_channels", 320)
        cfg = _ModelConfig.config[num_pool_layers]
        tf.logging.info("Model config: {}".format(kwargs))

        with tf.variable_scope(self.name):
            if hasattr(self.args, "img_grad") and self.args.img_grad:
                dz, dy, dx = tf.image.image_gradients(self._inputs["images"])
                inputs = tf.concat((self._inputs["images"], dz, dy, dx), axis=-1)
            else:
                inputs = self._inputs["images"]
            if self.use_spatial:
                inputs = tf.concat((self._inputs["images"], self._inputs["sp_guide"]), axis=-1)

            x = inputs
            c = init_channels
            end_pts = {}
            for block_key, block in cfg.items():
                with tf.variable_scope(block_key):
                    if block_key.startswith("conv_e") or block_key == "bridge":
                        for layer_key, layer in block.items():
                            x = slim.conv3d(x, c, kernel_size=layer["kernel"], stride=layer["stride"], scope=layer_key)
                        end_pts[block_key] = {"x": x, "c": c}
                        c = min(c * 2, max_channels)
                    elif block_key.startswith("conv_d"):
                        for layer_key, layer in block.items():
                            if layer_key == "up":
                                enc_key = block_key.replace("d", "e")
                                c = end_pts[enc_key]["c"]
                                x = slim.conv3d_transpose(x, c, kernel_size=layer["kernel"], stride=layer["stride"],
                                                          biases_initializer=None, scope=layer_key)
                                x = tf.concat([end_pts[enc_key]["x"], x], axis=-1)
                            else:
                                x = slim.conv3d(x, c, kernel_size=layer["kernel"], stride=layer["stride"], scope=layer_key)

            logits = slim.conv3d(x, self.num_classes, kernel_size=1, normalizer_fn=None, activation_fn=None, scope="logits")
            self._layers["logits"] = logits

            # Probability & Prediction
            self.ret_prob = kwargs.get("ret_prob", False)
            self.ret_pred = kwargs.get("ret_pred", False)
            if self.ret_prob or self.ret_pred:
                with tf.name_scope("Prediction"):
                    self.probability = slim.softmax(logits)
                    split = tf.split(self.probability, self.num_classes, axis=-1)
                    if self.ret_prob:
                        for i in range(1, self.num_classes):
                            self.predictions[self.classes[i] + "Prob"] = split[i]
                    if self.ret_pred:
                        zeros = tf.zeros_like(split[0], dtype=tf.uint8)
                        ones = tf.ones_like(zeros, dtype=tf.uint8)
                        for i in range(1, self.num_classes):
                            obj = self.classes[i] + "Pred"
                            self.predictions[obj] = tf.where(split[i] > 0.5, ones, zeros, name=obj)
                            self._image_summaries[obj] = self.predictions[obj]

    def _build_loss(self):
        with tf.name_scope("Losses/"):
            self._inputs["labels"].set_shape([self.bs, self.depth, self.height, self.width])
            w_param = self._get_weights_params()
            has_loss = False
            if "xentropy" in self.args.loss_type:
                losses.weighted_sparse_softmax_cross_entropy(logits=self._layers["logits"],
                                                             labels=self._inputs["labels"],
                                                             w_type=self.args.loss_weight_type, **w_param)
                has_loss = True
            if not has_loss:
                raise ValueError("Not supported loss_type: {}".format(self.args.loss_type))

            total_loss = tf.losses.get_total_loss()
        return total_loss

    def _build_metrics(self):
        if not self.ret_pred:
            tf.logging.warning("Model not return prediction, no metric will be created! "
                               "If needed, set ret_pred=true in <model>.yml")
            return

        with tf.name_scope("Metrics"):
            with tf.name_scope("LabelProcess/"):
                one_hot_label = tf.one_hot(self._inputs["labels"], self.num_classes)
                split_labels = tf.split(one_hot_label, self.num_classes, axis=-1)
            for i in range(1, self.num_classes):
                obj = self.classes[i]
                logits = self.predictions[obj + "Pred"]
                labels = split_labels[i]
                for met in self.args.metrics_train:
                    metric_func = eval("metrics.metric_" + met.lower())
                    res = metric_func(logits, labels, name=obj + met, reduce=True)
                    # "{}/{}" format will be recognized by estimator and printed at each display step
                    self.metrics_dict["{}/{}".format(obj, met)] = res

    def _build_summaries(self):
        if self.mode == ModeKeys.TRAIN:
            # Make sure all the elements are positive
            with tf.name_scope("SumImage"):
                if self.args.im_channel == 1:
                    image = self._inputs["images"][:, self.depth // 2]
                    images = [image - tf.reduce_min(image)]
                elif self.args.im_channel == 2:
                    image, res2d = tf.split(self._inputs["images"][:, self.depth // 2], 2, axis=-1)
                    images = [image - tf.reduce_min(image), res2d]

            for i, image in enumerate(images):
                tf.summary.image("{}/Image{}".format(self.args.tag, i), image,
                                 max_outputs=1, collections=[self.DEFAULT])

            with tf.name_scope("SumLabel"):
                labels = tf.expand_dims(self._inputs["labels"][:, self.depth // 2], axis=-1)
                labels_uint8 = tf.cast(labels * 255 / len(self.args.classes), tf.uint8)
            tf.summary.image("{}/Label".format(self.args.tag), labels_uint8,
                             max_outputs=1, collections=[self.DEFAULT])

            for key, value in self._image_summaries.items():
                tf.summary.image("{}/{}".format(self.args.tag, key), value[:, self.depth // 2] * 255,
                                 max_outputs=1, collections=[self.DEFAULT])
