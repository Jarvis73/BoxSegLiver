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
import tensorflow.contrib.slim as slim

import loss_metrics as losses
from Networks import base
from utils import distribution_utils

ModeKeys = tf.estimator.ModeKeys
metrics = losses


def _check_size_type(size):
    if size < 0:
        return None
    return size


class UNet(base.BaseNet):
    def __init__(self, args, name=None):
        """ Don't create ops/tensors in __init__() """
        super(UNet, self).__init__(args)
        self.name = name or "UNet"
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
                            biases_regularizer=default_b_regu) as scope:
            if self.args.without_norm:
                return scope
            normalizer, params = self._get_normalization()
            with slim.arg_scope([slim.conv2d],
                                normalizer_fn=normalizer,
                                normalizer_params=params) as scope2:
                return scope2

    def _build_network(self, *args, **kwargs):
        out_channels = kwargs.get("init_channels", 64)
        num_down_samples = kwargs.get("num_down_samples", 4)

        # Tensorflow can not infer input tensor shape when constructing graph
        self.height = _check_size_type(self.height)
        self.width = _check_size_type(self.width)
        self._inputs["images"].set_shape([self.bs, self.height, self.width, self.channel])
        self._inputs["labels"].set_shape([self.bs, None, None])

        tensor_out = self._inputs["images"]
        down_sample = kwargs.get("input_down_sample", False)
        h, w = self._inputs["images"].shape[1:3]
        if down_sample:
            rate = kwargs.get("input_down_sample_rate", 2)
            tensor_out = tf.image.resize_bilinear(tensor_out, [h // rate, w // rate])

        with tf.variable_scope(self.name, "UNet"):
            encoder_layers = {}

            # encoder
            for i in range(num_down_samples):
                with tf.variable_scope("Encode{:d}".format(i + 1)):
                    tensor_out = slim.repeat(tensor_out, 2, slim.conv2d, out_channels, 3)   # Conv-BN-ReLU
                    encoder_layers["Encode{:d}".format(i + 1)] = tensor_out
                    tensor_out = slim.max_pool2d(tensor_out, [2, 2])
                out_channels *= 2

            # Encode-Decode-Bridge
            tensor_out = slim.repeat(tensor_out, 2, slim.conv2d, out_channels, 3, scope="ED-Bridge")

            # decoder
            for i in reversed(range(num_down_samples)):
                out_channels /= 2
                with tf.variable_scope("Decode{:d}".format(i + 1)):
                    tensor_out = slim.conv2d_transpose(tensor_out,
                                                       tensor_out.get_shape()[-1] // 2, 2, 2)
                    tensor_out = tf.concat((encoder_layers["Encode{:d}".format(i + 1)], tensor_out), axis=-1)
                    tensor_out = slim.repeat(tensor_out, 2, slim.conv2d, out_channels, 3)

            # final
            with slim.arg_scope([slim.conv2d],
                                activation_fn=None,
                                normalizer_fn=None, normalizer_params=None):
                logits = slim.conv2d(tensor_out, self.num_classes, 1, scope="AdjustChannels")
                if down_sample:
                    logits = tf.image.resize_bilinear(logits, [h, w])
                self._layers["logits"] = logits

            # Probability & Prediction
            self.ret_prob = kwargs.get("ret_prob", False)
            self.ret_pred = kwargs.get("ret_pred", False)
            if self.ret_prob or self.ret_pred:
                self.probability = slim.softmax(logits)
                split = tf.split(self.probability, self.num_classes, axis=-1)
                if self.ret_prob:
                    for i in range(1, self.num_classes):
                        self._layers[self.classes[i] + "Prob"] = split[i]
                if self.ret_pred:
                    zeros = tf.zeros_like(split[0], dtype=tf.uint8)
                    ones = tf.ones_like(zeros, dtype=tf.uint8)
                    for i in range(1, self.num_classes):
                        obj = self.classes[i] + "Pred"
                        self._layers[obj] = tf.where(split[i] > 0.5, ones, zeros, name=obj)
                        # if self.args.only_tumor and self.classes[i] == "Tumor":
                        #     self._layers[obj] = self._layers[obj] * tf.cast(
                        #         tf.expand_dims(self._inputs["livers"], axis=-1), tf.uint8)
                        self._image_summaries[obj] = self._layers[obj]
        return

    def _build_loss(self):
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

        with tf.name_scope("Losses/"):
            total_loss = tf.losses.get_total_loss()
            tf.losses.add_loss(total_loss)
            return total_loss

    def _build_metrics(self):
        if not self.ret_pred:
            tf.logging.warning("Model not return prediction, no metric will be created! "
                               "If needed, set ret_pred=true in <model>.yml")
            return
        if self.mode in [ModeKeys.TRAIN, ModeKeys.PREDICT]:
            with tf.name_scope("LabelProcess/"):
                one_hot_label = tf.one_hot(self._inputs["labels"], self.num_classes)
                split_labels = tf.split(one_hot_label, self.num_classes, axis=-1)

            with tf.name_scope("Metrics"):
                for i in range(1, self.num_classes):
                    obj = self.classes[i]
                    logits = self._layers[obj + "Pred"]
                    labels = split_labels[i]
                    for met in self.args.metrics_train:
                        metric_func = eval("metrics.metric_" + met.lower())
                        res = metric_func(logits, labels, name=obj + met, reduce=self.is_training)
                        self.metrics_dict["{}/{}".format(obj, met)] = res

    def _build_summaries(self):
        if self.mode == ModeKeys.TRAIN:
            # Make sure all the elements are positive
            images = []
            with tf.name_scope("SumImage"):
                if self.args.im_channel == 1:
                    images.append(self._inputs["images"])
                elif self.args.im_channel == 2:
                    image1 = self._inputs["images"][..., 0:1]
                    images.append(image1 - tf.reduce_min(image1))
                    image2 = self._inputs["images"][..., 1:2]
                    images.append(image2 - tf.reduce_min(image2))
                elif self.args.im_channel == 3:
                    images = self._inputs["images"][..., 1:2]
                    images = [images - tf.reduce_min(images)]
                elif self.args.im_channel == 4:
                    image1 = self._inputs["images"][..., 1:2]
                    images.append(image1 - tf.reduce_min(image1))
                    image2 = self._inputs["images"][..., 3:4]
                    images.append(image2 - tf.reduce_min(image2))

            for image in images:
                tf.summary.image("{}/{}".format(self.args.tag, image.op.name), image,
                                 max_outputs=1, collections=[self.DEFAULT])

            with tf.name_scope("SumLabel"):
                labels = tf.expand_dims(self._inputs["labels"], axis=-1)
                labels_uint8 = tf.cast(labels * 255 / len(self.args.classes), tf.uint8)
                # if self.args.only_tumor:
                #     livers_uint8 = tf.cast(tf.expand_dims(self._inputs["livers"], axis=-1)
                #                            * 255 / len(self.args.classes), tf.uint8)
                #     tf.summary.image("{}/Liver".format(self.args.tag), livers_uint8,
                #                      max_outputs=1, collections=[self.DEFAULT])
            tf.summary.image("{}/{}".format(self.args.tag, labels.op.name), labels_uint8,
                             max_outputs=1, collections=[self.DEFAULT])

            for key, value in self._image_summaries.items():
                tf.summary.image("{}/{}".format(self.args.tag, key), value * 255,
                                 max_outputs=1, collections=[self.DEFAULT])

            for tensor in losses.get_losses():
                tf.summary.scalar("{}/{}".format(self.args.tag, tensor.op.name), tensor,
                                  collections=[self.DEFAULT])

            for tensor in metrics.get_metrics():
                tf.summary.scalar("{}/{}".format(self.args.tag, tensor.op.name), tensor,
                                  collections=[self.DEFAULT])

        return
