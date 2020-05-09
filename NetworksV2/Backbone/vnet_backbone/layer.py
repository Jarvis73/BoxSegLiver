import tensorflow as tf
import tensorflow.contrib.slim as slim


def get_num_channels(x):
    """
    :param x: an input tensor with shape [batch_size, ..., num_channels]
    :return: the number of channels of x
    """
    return int(x.get_shape()[-1])


def resnet_add(x1, x2):
    x1_shape = x1.get_shape().as_list()
    x2_shape = x2.get_shape().as_list()
    if x1_shape[-1] != x2_shape[-1]:
        pad = [(0, 0) for _ in range(len(x1_shape) - 1)] + [(0, x1_shape[-1] - x2_shape[-1])]
        residual_connection = x1 + tf.pad(x2, pad)
    else:
        residual_connection = x1 + x2
    return residual_connection


def conv_bn_relu_drop(input_, num_outputs, kernel, strides=None, activation_type=None, norm_type=None, is_train=True,
                      keep_prob=1, replace=1, resnet=True):
    with tf.variable_scope('conv_bn_relu_drop'):
        x = input_
        for i in range(replace):
            x = slim.conv3d(x, num_outputs, kernel_size=kernel, stride=strides, scope='conv_' + str(i + 1))
            x = tf.nn.dropout(x, keep_prob=keep_prob)
            if resnet and i == replace - 1:
                x = resnet_add(x, input_)
        return x


def concat_conv_bn_relu_drop(input_, feature, num_outputs, kernel, strides, activation_type=None, norm_type=None,
                             is_train=True, keep_prob=1, replace=1, resnet=True):
    with tf.variable_scope('concat_conv_bn_relu_drop'):
        x = tf.concat([feature, input_], axis=-1)
        x = slim.instance_norm(x)
        for i in range(replace):
            if i == 0:
                x = slim.conv3d(x, num_outputs, kernel_size=kernel, stride=strides, scope='conv_' + str(i + 1))
                input_ = slim.instance_norm(x, scope='input')
            else:
                x = slim.conv3d(x, num_outputs, kernel_size=kernel, stride=strides, scope='conv_' + str(i + 1))
            x = tf.nn.dropout(x, keep_prob=keep_prob)
            if resnet and i == replace - 1:
                x = resnet_add(x, input_)
        return x
