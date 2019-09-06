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


class DenseUNet(base.BaseNet):
    def __init__(self, args, name=None):
        """ Don't create ops/tensors in __init__() """
        super(DenseUNet, self).__init__(args)
        self.name = name or "DenseUNet"
        self.classes.extend(self.args.classes)

        self.bs = distribution_utils.per_device_batch_size(args.batch_size, args.num_gpus)
        self.height = args.im_height
        self.width = args.im_width
        self.channel = args.im_channel
        self.dropout = args.dropout

    def _net_arg_scope(self, *args, **kwargs):
        default_w_regu, _ = self._get_regularizer()
        default_w_init, _ = self._get_initializer()

        with slim.arg_scope([slim.conv2d],
                            weights_regularizer=default_w_regu,
                            weights_initializer=default_w_init,
                            biases_regularizer=None,
                            biases_initializer=None,
                            activation_fn=None,
                            normalizer_fn=None) as scope:
            return scope

    def _build_network(self, *args, **kwargs):
        # Tensorflow can not infer input tensor shape when constructing graph
        self.height = base._check_size_type(self.height)
        self.width = base._check_size_type(self.width)
        self._inputs["images"].set_shape([self.bs, self.height, self.width, self.channel])
        if "labels" in self._inputs:
            self._inputs["labels"].set_shape([self.bs, None, None])

        x = self._inputs["images"]

        with tf.variable_scope(self.name, "DenseUNet"):
            nb_filter = 96
            nb_layers = [6, 12, 36, 24]
            nb_dense_block = 4
            growth_rate = 48
            compression = 0.5
            box = []

            x = slim.conv2d(x, nb_filter, 7, 2, scope="conv1")
            x = slim.batch_norm(x, scale=True, epsilon=1e-5, activation_fn=tf.nn.relu, scope="conv1_bn")
            box.append(x)
            x = slim.max_pool2d(x, 3, 2, padding="SAME", scope="pool1")

            for block_idx in range(nb_dense_block - 1):
                stage = block_idx + 2
                x, nb_filter = self.dense_block(x, stage, nb_layers[block_idx], nb_filter, growth_rate,
                                                dropout_rate=self.dropout)
                box.append(x)
                x = self.transition_block(x, stage, nb_filter, compression=compression, dropout_rate=self.dropout)
                nb_filter = int(nb_filter * compression)
            final_stage = stage + 1
            x, nb_filter = self.dense_block(x, final_stage, nb_layers[-1], nb_filter, growth_rate,
                                            dropout_rate=self.dropout)
            x = slim.batch_norm(x, scale=True, epsilon=1e-5, activation_fn=tf.nn.relu,
                                scope='conv'+str(final_stage)+'_blk_bn')
            box.append(x)

            up0 = tf.image.resize_nearest_neighbor(x, tf.shape(x)[1:3] * 2, align_corners=True)
            line0 = slim.conv2d(box[3], 2208, 1, weights_initializer="normal", scope="line0")
            up0_sum = line0 + up0
            conv_up0 = slim.conv2d(up0_sum, 768, 3, weights_initializer="normal", scope="conv_up0")
            bn_up0 = slim.batch_norm(conv_up0, scale=True, activation_fn=tf.nn.relu, scope='bn_up0')

            up1 = tf.image.resize_nearest_neighbor(bn_up0, tf.shape(bn_up0)[1:3] * 2, align_corners=True)
            up1_sum = box[2] + up1
            conv_up1 = slim.conv2d(up1_sum, 384, 3, weights_initializer="normal", scope="conv_up1")
            bn_up1 = slim.batch_norm(conv_up1, scale=True, activation_fn=tf.nn.relu, scope='bn_up1')

            up2 = tf.image.resize_nearest_neighbor(bn_up1, tf.shape(bn_up1)[1:3] * 2, align_corners=True)
            up2_sum = box[1] + up2
            conv_up2 = slim.conv2d(up2_sum, 96, 3, weights_initializer="normal", scope="conv_up2")
            bn_up2 = slim.batch_norm(conv_up2, scale=True, activation_fn=tf.nn.relu, scope='bn_up2')

            up3 = tf.image.resize_nearest_neighbor(bn_up2, tf.shape(bn_up2)[1:3] * 2, align_corners=True)
            up3_sum = box[0] + up3
            conv_up3 = slim.conv2d(up3_sum, 96, 3, weights_initializer="normal", scope="conv_up3")
            bn_up3 = slim.batch_norm(conv_up3, scale=True, activation_fn=tf.nn.relu, scope='bn_up3')

            up4 = tf.image.resize_nearest_neighbor(bn_up3, tf.shape(bn_up3)[1:3] * 2, align_corners=True)
            conv_up4 = slim.conv2d(up4, 64, 3, weights_initializer="normal", scope="conv_up4")
            conv_up4 = slim.dropout(conv_up4, keep_prob=0.7)
            bn_up4 = slim.batch_norm(conv_up4, scale=True, activation_fn=tf.nn.relu, scope='bn_up4')

            logits = slim.conv2d(bn_up4, self.num_classes, 1, weights_initializer="normal", scope="AdjustChannels")
            self._layers["logits"] = logits

            # Probability & Prediction
            self.ret_prob = kwargs.get("ret_prob", False)
            self.ret_pred = kwargs.get("ret_pred", False)
            if self.ret_prob or self.ret_pred:
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

    def conv_block(self, x, stage, branch, nb_filter, dropout_rate=None):
        conv_name_base = 'conv' + str(stage) + '_' + str(branch)
        inter_channel = nb_filter * 4
        x = slim.batch_norm(x, scale=True, epsilon=1e-5, activation_fn=tf.nn.relu, scope=conv_name_base + "_x1_bn")
        x = slim.conv2d(x, inter_channel, 1, scope=conv_name_base + "_x1")
        if dropout_rate:
            x = slim.dropout(x, keep_prob=1 - dropout_rate)
        x = slim.batch_norm(x, scale=True, epsilon=1e-5, activation_fn=tf.nn.relu, scope=conv_name_base + "_x2_bn")
        x = slim.conv2d(x, nb_filter, 3, scope=conv_name_base + "_x2")
        if dropout_rate:
            x = slim.dropout(x, keep_prob=1 - dropout_rate)
        return x

    def transition_block(self, x, stage, nb_filter, compression=1.0, dropout_rate=None):
        conv_name_base = 'conv' + str(stage) + '_blk'
        pool_name_base = 'pool' + str(stage)
        x = slim.batch_norm(x, scale=True, epsilon=1e-5, activation_fn=tf.nn.relu, scope=conv_name_base + "_bn")
        x = slim.conv2d(x, nb_filter * compression, 1, scope=conv_name_base)
        if dropout_rate:
            x = slim.dropout(x, keep_prob=1 - dropout_rate)
        x = slim.avg_pool2d(x, 2, padding="SAME", scope=pool_name_base)
        return x

    def dense_block(self, x, stage, nb_layers, nb_filter, growth_rate, dropout_rate=None, grow_nb_filters=True):
        concat_feat = x
        for i in range(nb_layers):
            branch = i + 1
            x = self.conv_block(concat_feat, stage, branch, growth_rate, dropout_rate)
            concat_feat = tf.concat((concat_feat, x), axis=-1, name="concat" + str(stage) + '_' + str(branch))
            if grow_nb_filters:
                nb_filter += growth_rate
        return concat_feat, nb_filter

    def _build_loss(self):
        with tf.name_scope("Losses/"):
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
