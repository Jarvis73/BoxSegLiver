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

import loss_metrics as losses
from NetworksV2 import base
from utils import distribution_utils

ModeKeys = tfes.estimator.ModeKeys
metrics = losses


class _ModelConfig(object):
    config = {}

    config[3] = {
        "image_e0": {"conv1": {"out":   32, "kernel": (3, 3), "stride": (1, 1), "dilated": 1},
                     "conv2": {"out":   32, "kernel": (3, 3), "stride": (1, 1), "dilated": 1}},
        "image_e1": {"conv1": {"out":   64, "kernel": (3, 3), "stride": (2, 2), "dilated": 1},
                     "conv2": {"out":   64, "kernel": (3, 3), "stride": (1, 1), "dilated": 1}},
        "image_e2": {"conv1": {"out":  128, "kernel": (3, 3), "stride": (2, 2), "dilated": 1},
                     "conv2": {"out":  128, "kernel": (3, 3), "stride": (1, 1), "dilated": 1}},

        "inter_e0": {"conv1": {"out":   32, "kernel": (3, 3), "stride": (1, 1), "dilated": 1},
                     "conv2": {"out":   32, "kernel": (3, 3), "stride": (1, 1), "dilated": 1}},
        "inter_e1": {"conv1": {"out":   64, "kernel": (3, 3), "stride": (2, 2), "dilated": 1},
                     "conv2": {"out":   64, "kernel": (3, 3), "stride": (1, 1), "dilated": 1}},
        "inter_e2": {"conv1": {"out":  128, "kernel": (3, 3), "stride": (2, 2), "dilated": 1},
                     "conv2": {"out":  128, "kernel": (3, 3), "stride": (1, 1), "dilated": 1}},

        "merge_e3": {"conv1": {"out":  512, "kernel": (3, 3), "stride": (2, 2), "dilated": 1},
                     "conv2": {"out":  512, "kernel": (3, 3), "stride": (1, 1), "dilated": 1},
                     "conv3": {"out": 1024, "kernel": (3, 3), "stride": (1, 1), "dilated": 2},
                     "conv4": {"out": 1024, "kernel": (3, 3), "stride": (1, 1), "dilated": 2}},

        "conv_d3":  {"conv1": {"out":  512, "kernel": (3, 3), "stride": (1, 1), "dilated": 2},
                     "conv2": {"out":  512, "kernel": (3, 3), "stride": (1, 1), "dilated": 1},
                     "conv3": {"out":  512, "kernel": (3, 3), "stride": (1, 1), "dilated": 1}},
        "conv_d2":  {"up":    {"out":  256, "kernel": (2, 2), "stride": (2, 2)},
                     "conv1": {"out":  256, "kernel": (3, 3), "stride": (1, 1), "dilated": 1},
                     "conv2": {"out":  256, "kernel": (3, 3), "stride": (1, 1), "dilated": 1}},
        "conv_d1":  {"up":    {"out":  128, "kernel": (2, 2), "stride": (2, 2)},
                     "conv1": {"out":  128, "kernel": (3, 3), "stride": (1, 1), "dilated": 1},
                     "conv2": {"out":  128, "kernel": (3, 3), "stride": (1, 1), "dilated": 1}},
        "conv_d0":  {"up":    {"out":   64, "kernel": (2, 2), "stride": (2, 2)},
                     "conv1": {"out":   64, "kernel": (3, 3), "stride": (1, 1), "dilated": 1},
                     "conv2": {"out":   64, "kernel": (3, 3), "stride": (1, 1), "dilated": 1}},

        "x":        "image_e2",
        "y":        "inter_e2",
    }


class InterUNet(base.BaseNet):
    def __init__(self, args, name=None):
        """ Don't create ops/tensors in __init__() """
        super(InterUNet, self).__init__(args)
        self.name = name or "SmallUNet"
        self.classes.extend(self.args.classes)

        self.bs = distribution_utils.per_device_batch_size(args.batch_size, args.num_gpus)
        self.height = args.im_height
        self.width = args.im_width
        self.channel = args.im_channel

    def _net_arg_scope(self, *args, **kwargs):
        default_w_regu, default_b_regu = self._get_regularizer()
        default_w_init, _ = self._get_initializer()

        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                            weights_regularizer=default_w_regu,
                            weights_initializer=default_w_init,
                            biases_regularizer=default_b_regu,
                            outputs_collections=["EPts"]) as scope:
            if self.args.without_norm:
                return scope
            normalizer, params = self._get_normalization()
            with slim.arg_scope([slim.conv2d],
                                normalizer_fn=normalizer,
                                normalizer_params=params) as scope2:
                return scope2

    def _build_network(self, *args, **kwargs):
        self.height = base._check_size_type(self.height)
        self.width = base._check_size_type(self.width)
        self._inputs["images"].set_shape([self.bs, self.height, self.width, self.channel])
        x = tf.concat((self._inputs["images"], self._inputs["sp_guide"]), axis=-1)
        y = self._inputs["images"]
        if self.args.img_grad:
            s = self.args.im_channel // 2
            sobel = tf.image.sobel_edges(self._inputs["images"][..., s:s + 1])
            sobel = tf.squeeze(sobel, axis=-2)
            y = tf.concat((y, sobel), axis=-1)
        z = None

        c = kwargs.get("init_channel_factor", 1)
        num_pool_layers = kwargs.get("num_pool_layers", 3)
        cfg = _ModelConfig.config[num_pool_layers]
        tf.logging.info("Model config: {}".format(kwargs))

        with tf.variable_scope(self.name):
            end_pts = {}
            for block_key, block in cfg.items():
                with tf.variable_scope(block_key):
                    if block_key.startswith("image_e"):
                        for layer_key, layer in block.items():
                            x = slim.conv2d(x, num_outputs=round(layer["out"] * c),
                                            kernel_size=layer["kernel"],
                                            stride=layer["stride"],
                                            rate=layer["dilated"],
                                            scope=layer_key)
                        end_pts[block_key] = x
                    elif block_key.startswith("inter_e"):
                        for layer_key, layer in block.items():
                            y = slim.conv2d(y, num_outputs=round(layer["out"] * c),
                                            kernel_size=layer["kernel"],
                                            stride=layer["stride"],
                                            rate=layer["dilated"],
                                            scope=layer_key)
                        end_pts[block_key] = y
                    elif block_key.startswith("merge_e"):
                        z = tf.concat([end_pts[cfg["x"]], end_pts[cfg["y"]]], axis=-1)
                        for layer_key, layer in block.items():
                            z = slim.conv2d(z, num_outputs=round(layer["out"] * c),
                                            kernel_size=layer["kernel"],
                                            stride=layer["stride"],
                                            rate=layer["dilated"],
                                            scope=layer_key)
                    elif block_key.startswith("conv_d"):
                        for layer_key, layer in block.items():
                            if layer_key == "up":
                                x_key = block_key.replace("d", "e").replace("conv", "image")
                                y_key = block_key.replace("d", "e").replace("conv", "inter")
                                z = slim.conv2d_transpose(z, num_outputs=round(layer["out"] * c),
                                                          kernel_size=layer["kernel"],
                                                          stride=layer["stride"],
                                                          biases_initializer=None,
                                                          scope=layer_key)
                                z = tf.concat([z, end_pts[x_key], end_pts[y_key]], axis=-1)
                            else:
                                z = slim.conv2d(z, num_outputs=round(layer["out"] * c),
                                                kernel_size=layer["kernel"],
                                                stride=layer["stride"],
                                                rate=layer["dilated"],
                                                scope=layer_key)

            logits = slim.conv2d(z, self.num_classes,
                                 kernel_size=1,
                                 normalizer_fn=None,
                                 activation_fn=None,
                                 scope="logits")
            self._layers["logits"] = logits

            # Probability & Prediction
            self.ret_prob = kwargs.get("ret_prob", False)
            self.ret_pred = kwargs.get("ret_pred", False)
            if self.ret_prob or self.ret_pred:
                self.probability = slim.softmax(logits)
                if self.ret_pred:
                    split = tf.split(self.probability, self.num_classes, axis=-1)
                    zeros = tf.zeros_like(split[0], dtype=tf.uint8)
                    ones = tf.ones_like(zeros, dtype=tf.uint8)
                    for i in range(1, self.num_classes):
                        obj = self.classes[i] + "Pred"
                        self.predictions[obj] = tf.where(split[i] > 0.5, ones, zeros, name=obj)
                        self._image_summaries[obj] = self.predictions[obj]

    def _build_loss(self):
        with tf.name_scope("Losses/"):
            self._inputs["labels"].set_shape([self.bs, self.height, self.width])
            w_param = self._get_weights_params()
            if self.args.loss_type == "xentropy":
                losses.weighted_sparse_softmax_cross_entropy(logits=self._layers["logits"],
                                                             labels=self._inputs["labels"],
                                                             w_type=self.args.loss_weight_type, **w_param)
            elif self.args.loss_type == "dice":
                losses.weighted_dice_loss(logits=self.probability,
                                          labels=self._inputs["labels"],
                                          w_type=self.args.loss_weight_type, **w_param)
            else:
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
        # Make sure all the elements are positive
        with tf.name_scope("SumImage"):
            if self.args.im_channel == 1:
                image = self._inputs["images"]
            elif self.args.im_channel == 3:
                image = self._inputs["images"][..., 1:2]
                image = image - tf.reduce_min(image)
        tf.summary.image("{}/{}".format(self.args.tag, "Source"), image,
                         max_outputs=1, collections=[self.DEFAULT])

        with tf.name_scope("SumLabel"):
            labels = tf.expand_dims(self._inputs["labels"], axis=-1)
            labels_uint8 = tf.cast(labels * 255 / len(self.args.classes), tf.uint8)
        tf.summary.image("{}/{}".format(self.args.tag, "Target"), labels_uint8,
                         max_outputs=1, collections=[self.DEFAULT])

        for key, value in self._image_summaries.items():
            tf.summary.image("{}/{}".format(self.args.tag, key), value * 255,
                             max_outputs=1, collections=[self.DEFAULT])
