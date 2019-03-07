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

from Networks import base
from Networks.Backbone import Xception
from Networks.Backbone import layers
import loss_metrics as losses

ModeKeys = tf.estimator.ModeKeys
metrics = losses

networks_map = {
    "xception_41": Xception.xception_41,
    "xception_42": Xception.xception_42,
}

networks_to_feature_maps = {
    "xception_41": {
        4: ["entry_flow/block2/unit_1/xception_module/"
            "separable_conv2_pointwise"],
    },
    "xception_42": {
        4: ["entry_flow/block2/unit_1/xception_module/"
            "separable_conv2_pointwise"],
    },
}

name_scope = {
    "xception_41": "xception_41",
    "xception_42": "xception_42"
}


def _check_size_type(size):
    if size < 0:
        return None
    return size


def _scale_dimension(dim, scale):
    """Scales the input dimension.

      Args:
        dim: Input dimension (a scalar or a scalar Tensor).
        scale: The amount of scaling applied to the input.

      Returns:
        Scaled dimension.
    """
    if isinstance(dim, tf.Tensor):
        return tf.cast((tf.to_float(dim) - 1.0) * scale + 1.0, dtype=tf.int32)
    else:
        return int((float(dim) - 1.0) * scale + 1.0)


class DeepLabV3Plus(base.BaseNet):
    def __init__(self, args, name=None):
        """ Don't create ops/tensors in __init__() """
        super(DeepLabV3Plus, self).__init__(args)
        self.name = name or "DeepLabV3p"
        self.classes.extend(self.args.classes)

        self.bs = args.batch_size
        self.height = args.im_height
        self.width = args.im_width
        self.channel = args.im_channel

        self.network = None

    def _net_arg_scope(self, *args, **kwargs):
        arg_scope = Xception.xception_arg_scope(
            weight_decay=self.args.weight_decay_rate,
            batch_norm_decay=0.9997,
            batch_norm_epsilon=1e-3,
            batch_norm_scale=True,
            regularize_depthwise=False,
            use_bounded_activation=False
        )

        with slim.arg_scope(arg_scope):
            with slim.arg_scope([slim.conv2d, slim.separable_conv2d],
                                weights_initializer=self._get_initializer()[0]) as scope:
                return scope

    def _tail_arg_scope(self):
        batch_norm_params = {
            'is_training': self.is_training and self.fine_tune_batch_norm,
            'decay': 0.9997,
            'epsilon': 1e-5,
            'scale': True,
        }
        with slim.arg_scope([slim.conv2d, slim.separable_conv2d],
                            weights_regularizer=slim.l2_regularizer(self.args.weight_decay_rate),
                            activation_fn=tf.nn.relu6 if self.use_bounded_activation else tf.nn.relu,
                            normalizer_fn=slim.batch_norm,
                            padding="SAME",
                            stride=1):
            with slim.arg_scope([slim.batch_norm], **batch_norm_params) as scope:
                return scope

    def _extract_features(self, images, **kwargs):
        output_stride = kwargs.get("output_stride", 8)
        features, end_points = networks_map[self.network](images,
                                                          is_training=self.is_training,
                                                          global_pool=False,
                                                          output_stride=output_stride,
                                                          multi_grid=None)
        with slim.arg_scope(self._tail_arg_scope()):
            depth = 256
            branch_logits = []

            if kwargs.get("add_image_level_feature", True):
                pool_height = _scale_dimension(self.height, 1.0 / output_stride)
                pool_width = _scale_dimension(self.width, 1.0 / output_stride)
                image_feature = slim.avg_pool2d(features, [pool_height, pool_width], [1, 1],
                                                padding="VALID")
                resize_height = _scale_dimension(self.height, 1.0 / output_stride)
                resize_width = _scale_dimension(self.width, 1.0 / output_stride)
                image_feature = slim.conv2d(image_feature, depth, 1, scope="image_pooling")
                image_feature = tf.image.resize_bilinear(image_feature, [resize_height, resize_width],
                                                         align_corners=True)
                image_feature.set_shape([self.args.batch_size, resize_height, resize_width, depth])
                branch_logits.append(image_feature)

            # Employ a 1x1 convolution.
            branch_logits.append(slim.conv2d(features, depth, 1, scope="aspp0"))

            if self.atrous_rate:
                # Employ 3x3 convolutions with different atrous rates.
                for i, rate in enumerate(self.atrous_rate):
                    scope = "aspp{}".format(i + 1)
                    if kwargs.get("aspp_with_separable_conv", True):
                        aspp_features = layers.split_separable_conv2d(
                            features,
                            filters=depth,
                            rate=rate,
                            weight_decay=self.args.weight_decay_rate,
                            scope=scope)
                    else:
                        aspp_features = slim.conv2d(
                            features, depth, 3, rate=rate, scope=scope)
                    branch_logits.append(aspp_features)

            concat_logits = tf.concat(branch_logits, 3)
            concat_logits = slim.conv2d(concat_logits, depth, 1, scope="concat_projection")
            concat_logits = slim.dropout(concat_logits,
                                         keep_prob=0.9,
                                         is_training=self.is_training,
                                         scope="concat_projection_dropout")
            return concat_logits, end_points

    def _refine_by_decoder(self, features, end_points):
        """Adds the decoder to obtain sharper segmentation results.
        Args:
          features: A tensor of size [batch, features_height, features_width,
            features_channels].
          end_points: A dictionary from components of the network to the corresponding
            activation.
        Returns:
          Decoder output with size [batch, decoder_height, decoder_width,
            decoder_channels].
        Raises:
          ValueError: If crop_size is None.
        """
        with slim.arg_scope(self._tail_arg_scope()):
            with tf.variable_scope("decoder", "decoder", [features]):
                decoder_features = features
                decoder_stage = 0
                scope_suffix = ""
                for output_stride in self.decoder_output_stride:
                    feature_list = networks_to_feature_maps[self.network][output_stride]
                    # If only one decoder stage, we do not change the scope name in
                    # order for backward compactibility.
                    if decoder_stage:
                        scope_suffix = "_{}".format(decoder_stage)
                    for i, name in enumerate(feature_list):
                        decoder_features_list = [decoder_features]
                        feature_name = "{}/{}/{}".format(self.name, name_scope[self.network], name)
                        decoder_features_list.append(slim.conv2d(
                            end_points[feature_name], 48, 1,
                            scope="feature_projection" + str(i) + scope_suffix))
                        decoder_height = _scale_dimension(self.height, 1.0 / output_stride)
                        decoder_width = _scale_dimension(self.width, 1.0 / output_stride)
                        # Resize to decoder_height/decoder_width.
                        for j, feature in enumerate(decoder_features_list):
                            decoder_features_list[j] = tf.image.resize_bilinear(
                                feature, [decoder_height, decoder_width], align_corners=True)
                            decoder_features_list[j].set_shape([self.bs, decoder_height, decoder_width, None])
                        decoder_depth = 256
                        if self.decoder_use_separable_conv:
                            decoder_features = layers.split_separable_conv2d(
                                tf.concat(decoder_features_list, 3),
                                filters=decoder_depth,
                                rate=1,
                                weight_decay=self.args.weight_decay_rate,
                                scope="decoder_conv0" + scope_suffix)
                            decoder_features = layers.split_separable_conv2d(
                                decoder_features,
                                filters=decoder_depth,
                                rate=1,
                                weight_decay=self.args.weight_decay_rate,
                                scope='decoder_conv1' + scope_suffix)
                        else:
                            num_convs = 2
                            decoder_features = slim.repeat(
                                tf.concat(decoder_features_list, 3),
                                num_convs,
                                slim.conv2d,
                                decoder_depth,
                                3,
                                scope="decoder_conv" + str(i) + scope_suffix)
                    decoder_stage += 1
                return decoder_features

    def _get_branch_logits(self,
                           features,
                           atrous_rates=None,
                           kernel_size=1,
                           scope_suffix=''):
        """Gets the logits from each model's branch.

        The underlying model is branched out in the last layer when atrous
        spatial pyramid pooling is employed, and all branches are sum-merged
        to form the final logits.

        Args:
          features: A float tensor of shape [batch, height, width, channels].
          atrous_rates: A list of atrous convolution rates for last layer.
          kernel_size: Kernel size for convolution.
          scope_suffix: Scope suffix for the model variables.

        Returns:
          Merged logits with shape [batch, height, width, num_classes].

        Raises:
          ValueError: Upon invalid input kernel_size value.
        """
        # When using batch normalization with ASPP, ASPP has been applied before
        # in extract_features, and thus we simply apply 1x1 convolution here.
        if self.aspp_with_batch_norm or atrous_rates is None:
            if kernel_size != 1:
                raise ValueError('Kernel size must be 1 when atrous_rates is None or '
                                 'using aspp_with_batch_norm. Gets %d.' % kernel_size)
            atrous_rates = [1]

        with slim.arg_scope(
                [slim.conv2d],
                weights_regularizer=slim.l2_regularizer(self.args.weight_decay_rate),
                weights_initializer=self._get_initializer()[0]):
            with tf.variable_scope("logits", "logits", [features]):
                branch_logits = []
                for i, rate in enumerate(atrous_rates):
                    scope = scope_suffix
                    if i:
                        scope += '_%d' % i

                    branch_logits.append(
                        slim.conv2d(
                            features,
                            self.num_classes,
                            kernel_size=kernel_size,
                            rate=rate,
                            activation_fn=None,
                            normalizer_fn=None,
                            scope=scope))

                return tf.add_n(branch_logits)

    def _build_network(self, *args, **kwargs):
        # Tensorflow can not infer input tensor shape when constructing graph
        self.height = _check_size_type(self.height)
        self.width = _check_size_type(self.width)
        self._inputs["images"].set_shape([self.bs, self.height, self.width, self.channel])
        self._inputs["labels"].set_shape([self.bs, None, None])

        self.network = kwargs.get("network", None)
        self.fine_tune_batch_norm = kwargs.get("fine_tune_batch_norm", True)
        self.use_bounded_activation = kwargs.get("use_bounded_activation", False)
        self.decoder_output_stride = kwargs.get("decoder_output_stride", [None])
        self.decoder_use_separable_conv = kwargs.get("decoder_use_separable_conv", True)
        self.atrous_rate = kwargs.get("atrous_rate", None)
        self.aspp_with_batch_norm = kwargs.get("aspp_with_batch_norm", False)

        with tf.variable_scope(self.name, "DeepLabV3p"):
            features, end_points = self._extract_features(self._inputs["images"], **kwargs)

            if self.decoder_output_stride:
                features = self._refine_by_decoder(features, end_points)
                features = self._get_branch_logits(features,
                                                   atrous_rates=self.atrous_rate,
                                                   kernel_size=1,
                                                   scope_suffix="semantic")

            if kwargs.get("upsample_logits", False):
                logits = tf.image.resize_bilinear(features, [self.height, self.width], align_corners=True,
                                                  name="final_logits")
            else:
                logits = tf.identity(features, name="final_logits")
            self._layers["logits"] = logits

            # Probability & Prediction
            self.ret_prob = kwargs.get("ret_prob", False)
            self.ret_pred = kwargs.get("ret_pred", False)
            if self.ret_prob or self.ret_pred:
                probability = slim.softmax(logits)
                split = tf.split(probability, self.num_classes, axis=-1)
                if self.ret_prob:
                    for i in range(1, self.num_classes):
                        self._layers[self.classes[i] + "Prob"] = split[i]
                if self.ret_pred:
                    zeros = tf.zeros_like(split[0], dtype=tf.uint8)
                    ones = tf.ones_like(zeros, dtype=tf.uint8)
                    for i in range(1, self.num_classes):
                        obj = self.classes[i] + "Pred"
                        self._layers[obj] = tf.where(split[i] > 0.5, ones, zeros, name=obj)
                        self._image_summaries[obj] = self._layers[obj]
        return

    def _build_loss(self):
        losses.weighted_sparse_softmax_cross_entropy(
            self._layers["logits"], self._inputs["labels"],
            self.args.loss_weight_type, name="Losses/", **self._get_weights_params())

        # Set the name of the total loss as "loss" which will be summarized by Estimator
        with tf.name_scope("Losses/"):
            total_loss = tf.losses.get_total_loss()
            tf.losses.add_loss(total_loss)
            return total_loss

    def _build_metrics(self):
        if not self.ret_pred:
            tf.logging.warning("Model not return prediction, no metric will be created! "
                               "If needed, set ret_pred=true in <model>.yml")
            return
        if self.mode in [ModeKeys.TRAIN, ModeKeys.EVAL]:
            with tf.name_scope("LabelProcess/"):
                graph = tf.get_default_graph()
                try:
                    one_hot_label = graph.get_tensor_by_name("LabelProcess/one_hot:0")
                except KeyError:
                    one_hot_label = tf.one_hot(self._inputs["labels"], self.num_classes)

                split_labels = []
                try:
                    for i in range(self.num_classes):
                        split_labels.append(graph.get_tensor_by_name("LabelProcess/split:{}".format(i)))
                except KeyError:
                    split_labels = tf.split(one_hot_label, self.num_classes, axis=-1)

            with tf.name_scope("Metrics"):
                for i in range(1, self.num_classes):
                    obj = self.classes[i]
                    logits = self._layers[obj + "Pred"]
                    labels = split_labels[i]
                    for met in self.args.metrics_train:
                        metric_func = eval("metrics.metric_" + met.lower())
                        metric_func(logits, labels, name=obj + met)

    def _build_summaries(self):
        if self.mode == ModeKeys.TRAIN:
            # Make sure all the elements are positive
            images = []
            if self.args.im_channel == 2:
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

            labels = tf.expand_dims(self._inputs["labels"], axis=-1)
            with tf.name_scope("LabelProcess/"):
                labels_uint8 = tf.cast(labels * 255 / len(self.args.classes), tf.uint8)
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
