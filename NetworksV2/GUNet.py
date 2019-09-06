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
from NetworksV2.Backbone import slim_nets
from utils import distribution_utils

ModeKeys = tfes.estimator.ModeKeys
metrics = losses


def _context_subnets(context,
                     mod_layers,
                     context_fc_channels,
                     init_channels,
                     num_down_samples,
                     dropout=None,
                     scope=None,
                     is_training=False,
                     context_model="fc",
                     context_conv_init_channels=16,
                     use_se=False):
    with tf.variable_scope(scope, "context", [context]):
        if use_se:
            n_modulator_param = context_fc_channels[-1] * sum(
                [1 for i in range(num_down_samples + 1) if i in mod_layers]) * 2
        else:
            n_modulator_param = init_channels * sum(
                [2 ** i for i in range(num_down_samples + 1) if i in mod_layers]) * 2

        if context_model == "fc":
            res = slim_nets.fc(context,
                               context_fc_channels + [n_modulator_param],
                               use_dropout=True if dropout else False,
                               keep_prob=1 - dropout,
                               is_training=is_training,
                               use_final_layer=True,
                               final_weight_initializer=tf.zeros_initializer(),
                               final_biases_initializer=tf.ones_initializer())
        elif context_model in ["vgg16B", "vgg16C", "vgg16D"]:
            res = slim_nets.vgg(context_model,
                                tf.expand_dims(context, axis=-1),
                                context_conv_init_channels,
                                context_fc_channels + [n_modulator_param],
                                slim.conv1d,
                                tf.layers.max_pooling1d,
                                use_dropout=True if dropout else False,
                                keep_prob=1 - dropout,
                                is_training=is_training,
                                use_fc=True,
                                use_final_layer=True,
                                final_weight_initializer=tf.zeros_initializer(),
                                final_biases_initializer=tf.ones_initializer())
        elif context_model == "resnet":
            raise NotImplementedError
        else:
            raise ValueError("Not supported context model")

        return res


def conditional_normalization(inputs, gamma, reuse=None, scope=None):
    with tf.variable_scope(scope, 'ConditionalNorm', [inputs, gamma], reuse=reuse):
        inputs_shape = inputs.get_shape()
        inputs_rank = inputs_shape.ndims

        if inputs_rank is None:
            raise ValueError('Inputs %s has undefined rank.' % inputs.name)
        if inputs_rank != 4:
            raise ValueError('Inputs %s is not a 4D tensor.' % inputs.name)
        params_shape = inputs_shape[-1:]
        gamma = tf.expand_dims(tf.expand_dims(gamma, 1), 1)
        if not params_shape.is_fully_defined():
            raise ValueError('Inputs %s has undefined last dimension %s.' % (
                inputs.name, params_shape))
        return inputs * gamma


def _spatial_subnets(sp_guide,
                     base_channels,
                     mod_layers,
                     num_down_samples,
                     activation_fn=tf.nn.relu,
                     normalizer_fn=None,
                     normalizer_params=None):
    with tf.variable_scope("spatial"):
        spatial_params = []
        gs = sp_guide
        with slim.arg_scope([slim.conv2d],
                            activation_fn=activation_fn,
                            normalizer_fn=normalizer_fn,
                            normalizer_params=normalizer_params,
                            padding="SAME"):
            for i in range(num_down_samples + 1):
                spatial_params.append(
                    slim.conv2d(gs, base_channels * 2 ** (i + 1), 1, scope="conv%d" % (i + 1))
                    if i in mod_layers else None)
                if i < num_down_samples:
                    gs = slim.avg_pool2d(gs, 2, scope="pool%d" % (i + 1))
    return spatial_params


def modulated_conv_block(self, net, repeat, channels, dilation=1, scope_id=0, density_mod_id=0,
                         density_modulation_params=None,
                         spatial_modulation_params=None,
                         density_modulation=False,
                         spatial_modulation=False,
                         after_affine=False,
                         dropout=None,
                         is_training=True,
                         use_se=False,
                         context_feature_length=None):
    spatial_mod_id = 0
    if use_se and context_feature_length is None:
        raise ValueError("`context_feature_length` must be specified when `use_se` is True")

    with tf.variable_scope("down_conv{}".format(scope_id)):
        for i in range(repeat):
            with tf.variable_scope("mod_conv{}".format(i + 1)):
                if density_modulation or spatial_modulation or self.args.without_norm:
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
                if density_modulation:
                    if use_se:
                        context_feature = tf.slice(density_modulation_params, [0, density_mod_id],
                                                   [-1, context_feature_length], name="de_params")
                        # Combine context guide with attention
                        out = tf.reduce_mean(net, axis=(1, 2), name="global_avg_pool")
                        out = tf.concat((out, context_feature), axis=-1)
                        out = slim.fully_connected(out, (channels + context_feature_length) // 4)
                        den_params = slim.fully_connected(out, channels, activation_fn=tf.nn.sigmoid)
                        net = conditional_normalization(net, den_params, scope="density")
                        density_mod_id += context_feature_length
                    else:
                        den_params = tf.slice(density_modulation_params, [0, density_mod_id], [-1, channels],
                                              name="de_params")
                        net = conditional_normalization(net, den_params, scope="density")
                        density_mod_id += channels
                if spatial_modulation:
                    sp_params = tf.slice(spatial_modulation_params,
                                         [0, 0, 0, spatial_mod_id], [-1, -1, -1, channels],
                                         name="sp_params")
                    net = tf.add(net, sp_params, name="guide")
                    spatial_mod_id += channels
                if after_affine:
                    net = slim_nets.affine(net)
                net = tf.nn.relu(net)
        return net, density_mod_id


class GUNet(base.BaseNet):
    def __init__(self, args, name=None):
        """ Don't create ops/tensors in __init__() """
        super(GUNet, self).__init__(args)
        self.name = name or "GUNet"
        self.classes.extend(self.args.classes)

        self.bs = distribution_utils.per_device_batch_size(args.batch_size, args.num_gpus)
        self.height = args.im_height
        self.width = args.im_width
        self.channel = args.im_channel
        self.use_context_guide = args.use_context
        self.use_spatial_guide = args.use_spatial
        self.side_dropout = args.side_dropout
        self.dropout = args.dropout
        self.use_se = args.use_se

    def _net_arg_scope(self, *args, **kwargs):
        default_w_regu, default_b_regu = self._get_regularizer()
        default_w_init, _ = self._get_initializer()

        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                            weights_regularizer=default_w_regu,
                            weights_initializer=default_w_init,
                            biases_regularizer=default_b_regu):
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

        base_channels = kwargs.get("init_channels", 64)
        num_down_samples = kwargs.get("num_down_samples", 4)
        mod_layers = kwargs.get("mod_layers", [])
        context_fc_channels = kwargs.get("context_fc_channels", [256])
        context_model = kwargs.get("context_model", "fc")
        context_conv_init_channels = kwargs.get("context_conv_init_channels", 16)
        norm_with_center = kwargs.get("norm_with_center", False)
        norm_with_scale = kwargs.get("norm_with_scale", False)
        after_affine = kwargs.get("after_affine", False)
        tf.logging.info("Model config: {}".format(kwargs))

        with tf.variable_scope(self.name):
            if self.use_context_guide:
                self._inputs["context"].set_shape([self.bs, None])
                context_params = _context_subnets(self._inputs["context"],
                                                  mod_layers,
                                                  context_fc_channels,
                                                  base_channels,
                                                  num_down_samples,
                                                  dropout=self.side_dropout,
                                                  is_training=self.is_training,
                                                  context_model=context_model,
                                                  context_conv_init_channels=context_conv_init_channels,
                                                  use_se=self.use_se)
            else:
                context_params = None

            if self.use_spatial_guide:
                self._inputs["sp_guide"].set_shape([self.bs, self.height, self.width, 1])
                norm_params = {"scale": True, "epsilon": 0.001}
                if self.args.normalizer == "batch_norm":
                    norm_params.update({"decay": 0.99, "is_training": self.is_training})
                spatial_params = _spatial_subnets(self._inputs["sp_guide"],
                                                  base_channels=base_channels,
                                                  mod_layers=mod_layers,
                                                  num_down_samples=num_down_samples,
                                                  activation_fn=tf.nn.relu,
                                                  normalizer_fn=self._get_normalization()[0],
                                                  normalizer_params=norm_params)
            else:
                spatial_params = [None] * (num_down_samples + 1)

            def encoder_arg_scope():
                if self.args.without_norm:
                    return slim.current_arg_scope()
                else:
                    encoder_norm_params = {
                        'center': True if norm_with_center and not after_affine else False,
                        'scale': True if norm_with_scale and not after_affine else False,
                    }
                    if self.args.normalizer == "batch_norm":
                        encoder_norm_params.update({
                            'decay': 0.99,
                            'is_training': self.is_training
                        })
                    with slim.arg_scope([slim.conv2d],
                                        normalizer_fn=self._get_normalization()[0],
                                        normalizer_params=encoder_norm_params) as scope:
                        return scope

            # Encoder
            with tf.variable_scope("Encode"), slim.arg_scope(encoder_arg_scope()):
                context_crop_id = 0
                nets = [self._inputs["images"]]
                for i in range(num_down_samples + 1):
                    net, context_crop_id = modulated_conv_block(
                        self, nets[-1], 2, base_channels * 2 ** i, scope_id=i + 1,
                        density_mod_id=context_crop_id,
                        density_modulation_params=context_params,
                        spatial_modulation_params=spatial_params[i],
                        density_modulation=self.use_context_guide and i in mod_layers,
                        spatial_modulation=self.use_spatial_guide and i in mod_layers,
                        after_affine=after_affine,
                        dropout=self.dropout,
                        is_training=self.is_training,
                        use_se=self.use_se,
                        context_feature_length=context_fc_channels[-1])
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
                    images.append(image2)
                if self.use_context_guide:
                    image3 = tf.expand_dims(self._inputs["context"], axis=-1)
                    # Gaussian blur
                    kernel = tf.exp(-tf.convert_to_tensor([1., 0., 1.]) / (2 * 1.5 * 1.5)) / (2 * 3.14159 * 1.5 * 1.5)
                    kernel = tf.expand_dims(tf.expand_dims(kernel / tf.reduce_sum(kernel), axis=-1), axis=-1)
                    image3 = tf.nn.conv1d(image3, kernel, 1, padding="SAME")
                    image3 = tf.expand_dims(image3, axis=0)
                    images.append(tf.image.resize_nearest_neighbor(
                        image3, tf.concat(([self.bs * 10], tf.shape(self._inputs["context"])[1:]), axis=0),
                        align_corners=True))

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
