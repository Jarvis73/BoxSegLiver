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


def _density_modulator(density_hist, mod_early_conv, density_channels, init_channels,
                       scope=None, is_training=False, with_conv=False, conv_init_num_outs=64):
    with tf.variable_scope(scope, "mod_density", [density_hist]):
        n_modulator_param = init_channels * (mod_early_conv + 2 + 4 + 8 + 16) * 2

        net = density_hist
        if with_conv:
            net = tf.expand_dims(net, axis=-1)
            net = slim.repeat(net, 2, slim.conv1d, conv_init_num_outs, 3, scope="conv1")
            net = tf.layers.max_pooling1d(net, 2, 2, padding="same")
            net = slim.repeat(net, 2, slim.conv1d, conv_init_num_outs * 2, 3, scope="conv2")
            net = tf.layers.max_pooling1d(net, 2, 2, padding="same")
            net = slim.repeat(net, 3, slim.conv1d, conv_init_num_outs * 4, 3, scope="conv3")
            net = tf.layers.max_pooling1d(net, 2, 2, padding="same")
            net = slim.repeat(net, 3, slim.conv1d, conv_init_num_outs * 8, 3, scope="conv4")
            net = tf.layers.max_pooling1d(net, 2, 2, padding="same")
            net = slim.repeat(net, 3, slim.conv1d, conv_init_num_outs * 16, 3, scope="conv5")
            net = tf.layers.max_pooling1d(net, 2, 2, padding="same")
            net = slim.conv1d(net, density_channels, 7, padding="VALID", scope="fc6")
            net = slim.dropout(net, 0.5, is_training=is_training, scope="dropout1")
            net = slim.conv1d(net, density_channels, 1, scope="fc7")
            net = slim.dropout(net, 0.5, is_training=is_training, scope="dropout2")
            modulator_params = slim.conv1d(net, n_modulator_param, 1,
                                           weights_initializer=tf.zeros_initializer(),
                                           biases_initializer=tf.ones_initializer(),
                                           activation_fn=None, normalizer_fn=None, scope="fc8")
            modulator_params = tf.squeeze(modulator_params, axis=1)
        else:
            net = slim.fully_connected(density_hist, density_channels, scope="fc1")
            net = slim.dropout(net, 0.5, is_training=is_training, scope="dropout1")
            net = slim.fully_connected(net, density_channels, scope="fc2")
            net = slim.dropout(net, 0.5, is_training=is_training, scope="dropout2")
            modulator_params = slim.fully_connected(net, n_modulator_param,
                                                    weights_initializer=tf.zeros_initializer(),
                                                    biases_initializer=tf.ones_initializer(),
                                                    activation_fn=None, normalizer_fn=None, scope="fc3")
        return modulator_params


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


def modulated_conv_block(self, net, repeat, channels, dilation=1, scope_id=0, density_mod_id=0,
                         density_modulation_params=None,
                         spatial_modulation_params=None,
                         density_modulation=False,
                         spatial_modulation=False):
    spatial_mod_id = 0

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
                if density_modulation:
                    den_params = tf.slice(density_modulation_params, [0, density_mod_id], [-1, channels],
                                          name="de_params")
                    net = conditional_normalization(net, den_params,
                                                    scope="density")
                    density_mod_id += channels
                if spatial_modulation:
                    sp_params = tf.slice(spatial_modulation_params,
                                         [0, 0, 0, spatial_mod_id], [-1, -1, -1, channels],
                                         name="sp_params")
                    net = tf.add(net, sp_params, name="guide")
                    spatial_mod_id += channels
                net = tf.nn.relu(net)
        return net, density_mod_id


class OSMNUNet(base.BaseNet):
    def __init__(self, args, name=None):
        """ Don't create ops/tensors in __init__() """
        super(OSMNUNet, self).__init__(args)
        self.name = name or "OSMNUNet"
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
        images, labels, density_hists, sp_guide = (self._inputs["images"], self._inputs["labels"],
                                                   self._inputs["density_hists"], self._inputs["sp_guide"])
        out_channels = kwargs.get("init_channels", 64)

        # Tensorflow can not infer input tensor shape when constructing graph
        self.height = _check_size_type(self.height)
        self.width = _check_size_type(self.width)
        images.set_shape([self.bs, self.height, self.width, self.channel])
        labels.set_shape([self.bs, None, None])
        sp_guide.set_shape([self.bs, self.height, self.width, 1])
        density_hists.set_shape([self.bs, self.args.hist_bins * 2])

        mod_early_conv = kwargs.get("mod_early_conv", False)
        use_density_modulator = kwargs.get("use_density_modulator", True)
        use_spatial_modulator = kwargs.get("use_spatial_modulator", True)
        density_channels = kwargs.get("density_channels", 1024)
        with_conv = kwargs.get("with_conv", False)
        conv_init_num_outs = kwargs.get("conv_init_num_outs", 64)

        with tf.variable_scope(self.name, "UNet"):
            # density modulator
            density_modulation_params = (_density_modulator(density_hists, mod_early_conv, density_channels,
                                                            out_channels, is_training=self.is_training,
                                                            with_conv=with_conv,
                                                            conv_init_num_outs=conv_init_num_outs)
                                         if use_density_modulator else None)
            num_mod_layers = [2, 2, 2, 2, 2]
            norm_params = {
                'scale': True,
                'epsilon': 0.001,
            }
            if self.args.normalizer == "batch_norm":
                norm_params.update({
                    'decay': 0.99,
                    'is_training': self.is_training
                })

            # spatial modulator
            with tf.variable_scope("mod_sp"):
                with slim.arg_scope([slim.conv2d],
                                    activation_fn=tf.nn.relu,
                                    normalizer_fn=self._get_normalization()[0],
                                    normalizer_params=norm_params,
                                    padding="SAME"):
                    ds_mask = sp_guide
                    if mod_early_conv:
                        conv1_att = slim.conv2d(ds_mask, out_channels * num_mod_layers[0], 1, scope="conv1")
                    else:
                        conv1_att = None
                    ds_mask = slim.avg_pool2d(ds_mask, 2, scope="pool1")
                    conv2_att = slim.conv2d(ds_mask, 128 * num_mod_layers[1], 1, scope="conv2")
                    ds_mask = slim.avg_pool2d(ds_mask, 2, scope="pool2")
                    conv3_att = slim.conv2d(ds_mask, 256 * num_mod_layers[2], 1, scope="conv3")
                    ds_mask = slim.avg_pool2d(ds_mask, 2, scope="pool3")
                    conv4_att = slim.conv2d(ds_mask, 512 * num_mod_layers[3], 1, scope="conv4")
                    ds_mask = slim.avg_pool2d(ds_mask, 2, scope="pool4")
                    conv5_att = slim.conv2d(ds_mask, 1024 * num_mod_layers[4], 1, scope="conv5")

            def encode_arg_scope():
                if self.args.without_norm:
                    return slim.current_arg_scope()
                else:
                    encode_norm_params = {
                        'center': False if use_spatial_modulator else True,  # Replace with spatial guide
                        'scale': False if use_density_modulator else True,  # Replace with density guide
                    }
                    if self.args.normalizer == "batch_norm":
                        encode_norm_params.update({
                            'decay': 0.99,
                            'is_training': self.is_training
                        })
                    with slim.arg_scope([slim.conv2d],
                                        normalizer_fn=self._get_normalization()[0],
                                        normalizer_params=encode_norm_params) as scope:
                        return scope

            # encoder
            density_mod_id = 0
            with tf.variable_scope("Encode"), slim.arg_scope(encode_arg_scope()):
                net_1, density_mod_id = modulated_conv_block(
                    self, images, 2, out_channels * 1, scope_id=1,
                    density_mod_id=density_mod_id,
                    density_modulation_params=density_modulation_params,
                    spatial_modulation_params=conv1_att,
                    density_modulation=use_density_modulator and mod_early_conv,
                    spatial_modulation=use_spatial_modulator and mod_early_conv)

                net_2 = slim.max_pool2d(net_1, 2, scope="pool1")
                net_2, density_mod_id = modulated_conv_block(
                    self, net_2, 2, out_channels * 2, scope_id=2,
                    density_mod_id=density_mod_id,
                    density_modulation_params=density_modulation_params,
                    spatial_modulation_params=conv2_att,
                    density_modulation=use_density_modulator,
                    spatial_modulation=use_spatial_modulator)

                net_3 = slim.max_pool2d(net_2, 2, scope="pool2")
                net_3, density_mod_id = modulated_conv_block(
                    self, net_3, 2, out_channels * 4, scope_id=3,
                    density_mod_id=density_mod_id,
                    density_modulation_params=density_modulation_params,
                    spatial_modulation_params=conv3_att,
                    density_modulation=use_density_modulator,
                    spatial_modulation=use_spatial_modulator)

                net_4 = slim.max_pool2d(net_3, 2, scope="pool3")
                net_4, density_mod_id = modulated_conv_block(
                    self, net_4, 2, out_channels * 8, scope_id=4,
                    density_mod_id=density_mod_id,
                    density_modulation_params=density_modulation_params,
                    spatial_modulation_params=conv4_att,
                    density_modulation=use_density_modulator,
                    spatial_modulation=use_spatial_modulator)

                net_5 = slim.max_pool2d(net_4, 2, scope="pool4")
                net_5, density_mod_id = modulated_conv_block(
                    self, net_5, 2, out_channels * 16, scope_id=5,
                    density_mod_id=density_mod_id,
                    density_modulation_params=density_modulation_params,
                    spatial_modulation_params=conv5_att,
                    density_modulation=use_density_modulator,
                    spatial_modulation=use_spatial_modulator)
                _ = density_mod_id

            # decoder
            with tf.variable_scope("Decode"):
                net_4r = slim.conv2d_transpose(net_5, net_5.get_shape()[-1] // 2, 2, 2, scope="up4")
                net_4r = tf.concat((net_4, net_4r), axis=-1)
                net_4r = slim.repeat(net_4r, 2, slim.conv2d, out_channels * 8, 3, scope="up_conv4")

                net_3r = slim.conv2d_transpose(net_4r, net_4r.get_shape()[-1] // 2, 2, 2, scope="up3")
                net_3r = tf.concat((net_3, net_3r), axis=-1)
                net_3r = slim.repeat(net_3r, 2, slim.conv2d, out_channels * 4, 3, scope="up_conv3")

                net_2r = slim.conv2d_transpose(net_3r, net_3r.get_shape()[-1] // 2, 2, 2, scope="up2")
                net_2r = tf.concat((net_2, net_2r), axis=-1)
                net_2r = slim.repeat(net_2r, 2, slim.conv2d, out_channels * 2, 3, scope="up_conv2")

                net_1r = slim.conv2d_transpose(net_2r, net_2r.get_shape()[-1] // 2, 2, 2, scope="up1")
                net_1r = tf.concat((net_1, net_1r), axis=-1)
                net_1r = slim.repeat(net_1r, 2, slim.conv2d, out_channels * 1, 3, scope="up_conv1")

            # final
            with slim.arg_scope([slim.conv2d],
                                activation_fn=None):
                logits = slim.conv2d(net_1r, self.num_classes, 1, activation_fn=None,
                                     normalizer_fn=None, normalizer_params=None, scope="AdjustChannels")
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
                        res = metric_func(logits, labels, name=obj + met)
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
                    images = [images - tf.reduce_min(images)]
                    image2 = self._inputs["sp_guide"]
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
