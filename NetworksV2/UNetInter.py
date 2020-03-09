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


def modulated_conv_block(net, repeat, channels, dilation=1, scope_id=0,
                         outputs_collections=None):
    with tf.variable_scope("down_conv{}".format(scope_id)):
        for i in range(repeat):
            with tf.variable_scope("mod_conv{}".format(i + 1)) as sc:
                net = slim.conv2d(net, channels, 3, rate=dilation)
                utils_lib.collect_named_outputs(outputs_collections, sc.name, net)
        return net


class UNetInter(base.BaseNet):
    def __init__(self, args, name=None):
        """ Don't create ops/tensors in __init__() """
        super(UNetInter, self).__init__(args)
        self.name = name or "UNetInter"
        self.classes.extend(self.args.classes)

        self.bs = distribution_utils.per_device_batch_size(args.batch_size, args.num_gpus)
        self.height = args.im_height
        self.width = args.im_width
        self.channel = args.im_channel
        self.use_spatial_guide = args.use_spatial

    def _net_arg_scope(self, *args, **kwargs):
        default_w_regu, default_b_regu = self._get_regularizer()
        default_w_init, _ = self._get_initializer()

        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                            weights_regularizer=default_w_regu,
                            weights_initializer=default_w_init,
                            biases_regularizer=default_b_regu,
                            outputs_collections=["EPts"]):
            with slim.arg_scope([slim.avg_pool2d, slim.max_pool2d],
                                padding="SAME") as scope:
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
        # Tensorflow can not infer input tensor shape when constructing graph
        self._inputs["images"].set_shape([self.bs, self.height, self.width, self.channel])
        self._inputs["sp_guide"].set_shape([self.bs, self.height, self.width, 2])
        image = tf.concat((self._inputs["images"], self._inputs["sp_guide"]), axis=-1)

        base_channels = kwargs.get("init_channels", 64)
        num_down_samples = kwargs.get("num_down_samples", 4)
        tf.logging.info("Model config: {}".format(kwargs))

        with tf.variable_scope(self.name):
            def encoder_arg_scope():
                if self.args.without_norm:
                    return slim.current_arg_scope()
                else:
                    encoder_norm_params = {
                        'center': True,
                        'scale': True,
                    }
                    if self.args.normalizer == "batch_norm":
                        encoder_norm_params.update({
                            'decay': 0.99,
                            'is_training': self.is_training
                        })
                    with slim.arg_scope([slim.conv2d],
                                        normalizer_fn=self._get_normalization()[0],
                                        normalizer_params=encoder_norm_params,
                                        outputs_collections=None) as scope:
                        return scope

            # Encoder
            with tf.variable_scope("Encode"), slim.arg_scope(encoder_arg_scope()):
                nets = [image]
                for i in range(num_down_samples + 1):
                    net = modulated_conv_block(
                        nets[-1], 2, base_channels * 2 ** i, scope_id=i + 1,
                        outputs_collections=["EPts"])
                    nets[-1] = net
                    if i < num_down_samples:
                        net = slim.max_pool2d(nets[-1], 2, scope="pool%d" % (i + 1))
                        nets.append(net)

            # decoder
            with tf.variable_scope("Decode"):
                net_r = nets[-1]
                for i in reversed(range(num_down_samples)):
                    net_r = slim.conv2d_transpose(net_r, net_r.get_shape()[-1] // 2, 2, 2,
                                                  scope="up%d" % (i + 1))
                    net_r = tf.concat((nets[i], net_r), axis=-1)
                    net_r = slim.repeat(net_r, 2, slim.conv2d, base_channels * 2 ** i, 3,
                                        scope="up_conv%d" % (i + 1))

            # final
            with slim.arg_scope([slim.conv2d], activation_fn=None):
                logits = slim.conv2d(net_r, self.num_classes, 1, activation_fn=None,
                                     normalizer_fn=None, scope="AdjustChannels")
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
            self._inputs["labels"].set_shape([self.bs, self.height, self.width])
            w_param = self._get_weights_params()
            has_loss = False
            if "xentropy" in self.args.loss_type:
                losses.weighted_sparse_softmax_cross_entropy(logits=self._layers["logits"],
                                                             labels=self._inputs["labels"],
                                                             w_type=self.args.loss_weight_type, **w_param)
                has_loss = True
            if "dice" in self.args.loss_type:
                losses.weighted_dice_loss(logits=self.probability,
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
            images = []
            with tf.name_scope("SumImage"):
                if self.args.im_channel == 1:
                    images.append(self._inputs["images"])
                elif self.args.im_channel == 3:
                    images = self._inputs["images"][..., 1:2]
                    images = [images - tf.reduce_min(images)]   # Random noise problem
                if self.use_spatial_guide and not self.args.local_enhance:
                    images.extend(tf.split(self._inputs["sp_guide"], 2, axis=-1))
            for image in images:
                tf.summary.image("{}/{}".format(self.args.tag, image.op.name), image,
                                 max_outputs=1, collections=[self.DEFAULT])

            with tf.name_scope("SumLabel"):
                labels = tf.cast(tf.expand_dims(self._inputs["labels"], axis=-1), tf.float32)
                if self.use_spatial_guide and self.args.local_enhance:
                    if self.args.guide_channel == 2:
                        fg, bg = tf.split(self._inputs["sp_guide"], 2, axis=-1)
                        labels = labels + fg - bg + 1
                    else:
                        labels = labels + self._inputs["sp_guide"] + 1
            tf.summary.image("{}/Label".format(self.args.tag), labels,
                             max_outputs=1, collections=[self.DEFAULT])

            for key, value in self._image_summaries.items():
                tf.summary.image("{}/{}".format(self.args.tag, key), value * 255,
                                 max_outputs=1, collections=[self.DEFAULT])

            for x in tf.global_variables():
                if "spatial" in x.op.name:
                    tf.summary.histogram(x.op.name.replace("GUNet/spatial", "sp"), x)
