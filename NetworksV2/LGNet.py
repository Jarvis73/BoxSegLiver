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


def _spatial_subnets(sp_guide, mod_layers):
    with tf.variable_scope("spatial"):
        spatial_params = [[], []]
        layer_c = [64, 128, 256, 512, 1024]
        with slim.arg_scope([slim.conv2d],
                            normalizer_fn=None,
                            normalizer_params=None,
                            activation_fn=tf.nn.leaky_relu):
            sg = sp_guide
            tmp = [0] + [l for l in mod_layers[0]]
            for i, l in enumerate(tmp[1:]):
                sg = slim.avg_pool2d(sg, 2 ** (l - tmp[i]), 2 ** (l - tmp[i]),
                                     scope="pool_e%d" % (l + 1)) if l > 0 else sg
                spatial_params[0].append(
                    slim.conv2d(sg, layer_c[l], kernel_size=1, scope="conv_e%d" % (l + 1)))

            sg = sp_guide
            tmp = [0] + [l for l in mod_layers[1]]
            for i, l in enumerate(tmp[1:]):
                sg = slim.avg_pool2d(sg, 2 ** (l - tmp[i]), 2 ** (l - tmp[i]),
                                     scope="pool_d%d" % (l + 1)) if l > 0 else sg
                spatial_params[1].append(
                    slim.conv2d(sg, layer_c[l], kernel_size=1, scope="conv_d%d" % (l + 1)))
    return spatial_params


def modulated_conv_block(self, net, repeat, channels, dilation=1, scope_id=0,
                         spatial_modulation_params=None,
                         spatial_modulation=False,
                         after_affine=False,
                         dropout=None,
                         is_training=True,
                         outputs_collections=None):
    spatial_mod_id = 0

    with tf.variable_scope("down_conv{}".format(scope_id)):
        for i in range(repeat):
            with tf.variable_scope("mod_conv{}".format(i + 1)) as sc:
                if spatial_modulation or self.args.without_norm:
                    net = slim.conv2d(net, channels, 3, rate=dilation, activation_fn=None)
                else:
                    norm_params = {}
                    if self.args.normalizer == "batch_norm":
                        norm_params.update({"scale": True, "is_training": self.is_training})
                    net = slim.conv2d(net, channels, 3, rate=dilation, activation_fn=None,
                                      normalizer_fn=self._get_normalization()[0],
                                      normalizer_params=norm_params)
                if i != repeat - 1 and dropout:
                    net = slim.dropout(net, keep_prob=1 - dropout, is_training=is_training)
                if spatial_modulation:
                    sp_params = tf.slice(spatial_modulation_params,
                                         [0, 0, 0, spatial_mod_id], [-1, -1, -1, channels],
                                         name="sp_params")
                    net = tf.add(net, sp_params, name="guide")
                    spatial_mod_id += channels
                if after_affine:
                    net = slim_nets.affine(net)
                net = tf.nn.relu(net)
                utils_lib.collect_named_outputs(outputs_collections, sc.name, net)
        return net


class LGNet(base.BaseNet):
    def __init__(self, args, name=None):
        """ Don't create ops/tensors in __init__() """
        super(LGNet, self).__init__(args)
        self.name = name or "LGNet"
        self.classes.extend(self.args.classes)

        self.bs = distribution_utils.per_device_batch_size(args.batch_size, args.num_gpus)
        self.height = args.im_height
        self.width = args.im_width
        self.channel = args.im_channel
        self.use_spatial_guide = args.use_spatial
        self.dropout = args.dropout

    def _net_arg_scope(self, *args, **kwargs):
        default_w_regu, default_b_regu = self._get_regularizer()
        default_w_init, _ = self._get_initializer()

        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                            weights_regularizer=default_w_regu,
                            weights_initializer=default_w_init,
                            biases_regularizer=default_b_regu,
                            outputs_collections=["EPts"]):
            with slim.arg_scope([slim.avg_pool2d, slim.max_pool2d], padding="SAME"):
                normalizer, params = self._get_normalization()
                with slim.arg_scope([slim.conv2d],
                                    kernel_size=3,
                                    normalizer_fn=normalizer,
                                    normalizer_params=params,
                                    activation_fn=None) as scope:
                    return scope

    def merge_guide_act(self, x, layer, sp_params, mod_layers):
        if self.use_spatial_guide and layer in mod_layers:
            x = x + sp_params[mod_layers.index(layer)]
        return tf.nn.relu(x)

    def _build_network(self, *args, **kwargs):
        self.height = base._check_size_type(self.height)
        self.width = base._check_size_type(self.width)
        # Tensorflow can not infer input tensor shape when constructing graph
        self._inputs["images"].set_shape([self.bs, self.height, self.width, self.channel])

        mod_layers = kwargs.get("mod_layers", [[0, 1], [1, 0]])
        tf.logging.info("Model config: {}".format(kwargs))

        with tf.variable_scope(self.name):
            if self.use_spatial_guide:
                self._inputs["sp_guide"].set_shape([self.bs, self.height, self.width, self.args.guide_channel])
                norm_params = {"scale": True, "epsilon": 0.001}
                if self.args.normalizer == "batch_norm":
                    norm_params.update({"decay": 0.99, "is_training": self.is_training})
                spatial_params = _spatial_subnets(self._inputs["sp_guide"], mod_layers=mod_layers)
            else:
                spatial_params = [[None] * mod_layers[0], [None] * mod_layers[1]]

            if hasattr(self.args, "img_grad") and self.args.img_grad:
                dy, dx = tf.image.image_gradients(self._inputs["images"])
                inputs = tf.concat((self._inputs["images"], dy, dx), axis=-1)
            else:
                inputs = self._inputs["images"]

            with tf.variable_scope("conv_e0"):
                conv_e0_1 = slim.conv2d(inputs, 64, activation_fn=tf.nn.relu, scope="conv1")
                conv_e0_2 = slim.conv2d(conv_e0_1, 64, scope="conv2")
                conv_e0_a = self.merge_guide_act(conv_e0_2, 0, spatial_params[0], mod_layers[0])
                pool_0 = slim.max_pool2d(conv_e0_a, 2, scope="pool")

            with tf.variable_scope("conv_e1"):
                conv_e1_1 = slim.conv2d(pool_0, 128, activation_fn=tf.nn.relu, scope="conv1")
                conv_e1_2 = slim.conv2d(conv_e1_1, 128, scope="conv2")
                conv_e1_a = self.merge_guide_act(conv_e1_2, 1, spatial_params[0], mod_layers[0])
                pool_1 = slim.max_pool2d(conv_e1_a, 2, scope="pool")

            with tf.variable_scope("conv_e2"):
                conv_e2_1 = slim.conv2d(pool_1, 256, activation_fn=tf.nn.relu, scope="conv1")
                conv_e2_2 = slim.conv2d(conv_e2_1, 256, scope="conv2")
                conv_e2_a = self.merge_guide_act(conv_e2_2, 2, spatial_params[0], mod_layers[0])
                pool_2 = slim.max_pool2d(conv_e2_a, 2, scope="pool")

            with tf.variable_scope("conv_e3"):
                conv_e3_1 = slim.conv2d(pool_2, 512, activation_fn=tf.nn.relu, scope="conv1")
                conv_e3_2 = slim.conv2d(conv_e3_1, 512, scope="conv2")
                conv_e3_a = self.merge_guide_act(conv_e3_2, 3, spatial_params[0], mod_layers[0])
                pool_3 = slim.max_pool2d(conv_e3_a, 2, scope="pool")

            with tf.variable_scope("ED-Bridge"):
                conv_e4_1 = slim.conv2d(pool_3, 1024, activation_fn=tf.nn.relu, scope="conv1")
                conv_e4_2 = slim.conv2d(conv_e4_1, 1024, scope="conv2")
                conv_e4_a = self.merge_guide_act(conv_e4_2, 4, spatial_params[0], mod_layers[0])

            with tf.variable_scope("conv_d3"):
                up_3 = slim.conv2d_transpose(conv_e4_a, 512, 2, 2, scope="up")
                concat_3 = tf.concat((conv_e3_a, up_3), axis=-1)
                conv_d3_1 = slim.conv2d(concat_3, 512, scope="conv1")
                conv_d3_a = self.merge_guide_act(conv_d3_1, 3, spatial_params[1], mod_layers[1])
                conv_d3_2 = slim.conv2d(conv_d3_a, 512, activation_fn=tf.nn.relu, scope="conv2")

            with tf.variable_scope("conv_d2"):
                up_2 = slim.conv2d_transpose(conv_d3_2, 256, 2, 2, scope="up")
                concat_2 = tf.concat((conv_e2_a, up_2), axis=-1)
                conv_d2_1 = slim.conv2d(concat_2, 256, scope="conv1")
                conv_d2_a = self.merge_guide_act(conv_d2_1, 2, spatial_params[1], mod_layers[1])
                conv_d2_2 = slim.conv2d(conv_d2_a, 256, activation_fn=tf.nn.relu, scope="conv2")

            with tf.variable_scope("conv_d1"):
                up_1 = slim.conv2d_transpose(conv_d2_2, 128, 2, 2, scope="up")
                concat_1 = tf.concat((conv_e1_a, up_1), axis=-1)
                conv_d1_1 = slim.conv2d(concat_1, 128, scope="conv1")
                conv_d1_a = self.merge_guide_act(conv_d1_1, 1, spatial_params[1], mod_layers[1])
                conv_d1_2 = slim.conv2d(conv_d1_a, 128, activation_fn=tf.nn.relu, scope="conv2")

            with tf.variable_scope("conv_d0"):
                up_0 = slim.conv2d_transpose(conv_d1_2, 64, 2, 2, scope="up")
                concat_0 = tf.concat((conv_e0_a, up_0), axis=-1)
                conv_d0_1 = slim.conv2d(concat_0, 64, scope="conv1")
                conv_d0_a = self.merge_guide_act(conv_d0_1, 0, spatial_params[1], mod_layers[1])
                conv_d0_2 = slim.conv2d(conv_d0_a, 64, activation_fn=tf.nn.relu, scope="conv2")

            logits = slim.conv2d(conv_d0_2, self.num_classes, kernel_size=1, normalizer_fn=None, scope="logits")
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
                if self.use_spatial_guide:
                    image2 = self._inputs["sp_guide"]
                    images.extend(tf.split(image2, 2, axis=-1) if self.args.guide_channel == 2 else image2)

            for image in images:
                tf.summary.image("{}/{}".format(self.args.tag, image.op.name), image,
                                 max_outputs=1, collections=[self.DEFAULT])

            with tf.name_scope("SumLabel"):
                labels = tf.expand_dims(self._inputs["labels"], axis=-1)
                labels_uint8 = tf.cast(labels * 255 / len(self.args.classes), tf.uint8)
            tf.summary.image("{}/{}".format(self.args.tag, labels.op.name), labels_uint8,
                             max_outputs=1, collections=[self.DEFAULT])

            for key, value in self._image_summaries.items():
                tf.summary.image("{}/{}".format(self.args.tag, key), value * 255,
                                 max_outputs=1, collections=[self.DEFAULT])

            for x in tf.global_variables():
                if "spatial" in x.op.name:
                    tf.summary.histogram(x.op.name.replace("GUNet/spatial", "sp"), x)
