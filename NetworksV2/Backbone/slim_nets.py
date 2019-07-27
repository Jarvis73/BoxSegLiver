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

"""
Tensorflow backbones implemented by TF-Slim module

Please control nonlinear and normalizer with slim.arg_scope outside

"""

import tensorflow as tf
import tensorflow.contrib.slim as slim


def mlp(inputs, out_channels,
        use_dropout=True,
        keep_prob=0.5,
        is_training=True,
        use_final_layer=True,
        final_weight_initializer=slim.xavier_initializer(),
        final_biases_initializer=tf.zeros_initializer(),
        num_base=0):

    net = inputs
    for i, channel in enumerate(out_channels[:-1]):
        net = slim.fully_connected(net, channel, scope="fc%d" % (num_base + i + 1))
        if use_dropout:
            net = slim.dropout(net, keep_prob, is_training=is_training,
                               scope="dropout%d" % (num_base + i + 1))

    if use_final_layer:
        net = slim.fully_connected(net, out_channels[-1],
                                   weights_initializer=final_weight_initializer,
                                   biases_initializer=final_biases_initializer,
                                   activation_fn=None,
                                   normalizer_fn=None,
                                   scope="fc%d" % (num_base + len(out_channels)))
    return net


def vgg16C(inputs, first_layer_channel, out_channels, conv_op, pool_op,
           use_dropout=True,
           keep_prob=0.5,
           is_training=True,
           use_fc=True,
           use_final_layer=True,
           final_weight_initializer=slim.xavier_initializer(),
           final_biases_initializer=tf.zeros_initializer()):
    net = inputs
    net = slim.repeat(net, 2, conv_op, first_layer_channel * 1, 3, scope="conv1")
    net = pool_op(net, 2, 2, padding="same")
    net = slim.repeat(net, 2, conv_op, first_layer_channel * 2, 3, scope="conv2")
    net = pool_op(net, 2, 2, padding="same")
    net = slim.repeat(net, 2, conv_op, first_layer_channel * 4, 3, scope="conv3")
    net = conv_op(net, first_layer_channel * 4, 1, scope="conv3_3")
    net = pool_op(net, 2, 2, padding="same")
    net = slim.repeat(net, 2, conv_op, first_layer_channel * 8, 3, scope="conv4")
    net = conv_op(net, first_layer_channel * 8, 1, scope="conv4_3")
    net = pool_op(net, 2, 2, padding="same")
    net = slim.repeat(net, 2, conv_op, first_layer_channel * 8, 3, scope="conv5")
    net = conv_op(net, first_layer_channel * 8, 1, scope="conv5_3")
    net = pool_op(net, 2, 2, padding="same")

    if use_fc:
        net = slim.flatten(net, scope="flatten")
        net = mlp(net, out_channels, use_dropout, keep_prob, is_training, use_final_layer,
                  final_weight_initializer, final_biases_initializer, num_base=5)

    return net


def vgg16D(inputs, first_layer_channel, out_channels, conv_op, pool_op,
           use_dropout=True,
           keep_prob=0.5,
           is_training=True,
           use_fc=True,
           use_final_layer=True,
           final_weight_initializer=slim.xavier_initializer(),
           final_biases_initializer=tf.zeros_initializer()):
    net = inputs
    net = slim.repeat(net, 2, conv_op, first_layer_channel * 1, 3, scope="conv1")
    net = pool_op(net, 2, 2, padding="same")
    net = slim.repeat(net, 2, conv_op, first_layer_channel * 2, 3, scope="conv2")
    net = pool_op(net, 2, 2, padding="same")
    net = slim.repeat(net, 3, conv_op, first_layer_channel * 4, 3, scope="conv3")
    net = pool_op(net, 2, 2, padding="same")
    net = slim.repeat(net, 3, conv_op, first_layer_channel * 8, 3, scope="conv4")
    net = pool_op(net, 2, 2, padding="same")
    net = slim.repeat(net, 3, conv_op, first_layer_channel * 8, 3, scope="conv5")
    net = pool_op(net, 2, 2, padding="same")

    if use_fc:
        net = slim.flatten(net, scope="flatten")
        net = mlp(net, out_channels, use_dropout, keep_prob, is_training, use_final_layer,
                  final_weight_initializer, final_biases_initializer, num_base=5)

    return net


def vgg(net_name, *args, **kwargs):
    model = eval(net_name)
    return model(*args, **kwargs)


# Set alias
fc = mlp
