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
from tensorflow.python.ops import init_ops

ModeKeys = tf.estimator.ModeKeys


class BaseNet(object):
    def __init__(self, args):
        # Private
        self._name = "Base"
        self._mode = ModeKeys.TRAIN
        self._args = args
        self._inputs = {}
        self._layers = {}
        self._image_summaries = {}
        self.classes = ["Background"]

        # Summary collections
        self.DEFAULT = tf.GraphKeys.SUMMARIES

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, new_name):
        if new_name and isinstance(new_name, str):
            self._name = new_name

    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, new_mode):
        if new_mode in [ModeKeys.TRAIN, ModeKeys.EVAL, ModeKeys.PREDICT]:
            self._mode = new_mode

    @property
    def is_training(self):
        return self.mode == ModeKeys.TRAIN

    @property
    def args(self):
        return self._args

    @property
    def num_classes(self):
        return len(self.classes)

    @property
    def layers(self):
        return self._layers

    def _net_arg_scope(self, *args, **kwargs):
        raise NotImplementedError

    def _build_summaries(self):
        raise NotImplementedError

    def _build_network(self, *args, **kwargs):
        raise NotImplementedError

    def _build_loss(self):
        raise NotImplementedError

    def _build_metrics(self):
        raise NotImplementedError

    def _get_regularizer(self):
        w_reg = slim.l2_regularizer(self.args.weight_decay_rate)
        b_reg = None if self.args.bias_decay else w_reg

        return w_reg, b_reg

    def _get_initializer(self):
        if self.args.weight_init == "trunc_norm":
            w_init = init_ops.truncated_normal_initializer(mean=0.0, stddev=0.01)
        elif self.args.weight_init == "xavier":
            w_init = slim.xavier_initializer()
        # elif self.args.weight_init == "rand_norm":
        #     w_init = init_ops.random_normal_initializer(mean=0.0, stddev=0.01)
        # elif self.args.weight_init == "variance_scale":
        #     w_init = slim.variance_scaling_initializer()
        else:
            raise ValueError("Not supported weight initializer: " + self.args.weight_init)

        b_init = init_ops.constant_initializer()

        return w_init, b_init

    def _get_normalization(self):
        if self.args.normalizer == "batch_norm":
            normalizer_params = {"scale": True, "is_training": self.is_training}
            normalizer = slim.batch_norm
        # elif cfg.MODEL.NORMALIZATION == "instance_norm":
        #     normalizer_params = {}
        #     normalizer = slim.instance_norm
        # elif cfg.MODEL.NORMALIZATION == "layer_norm":
        #     normalizer_params = {}
        #     normalizer = slim.layer_norm
        # elif cfg.MODEL.NORMALIZATION == "group_norm":
        #     normalizer_params = {}
        #     normalizer = slim.group_norm
        else:
            raise ValueError("Not supported normalization function: " + self.args.normalizer)

        return normalizer, normalizer_params

    def _get_weights_params(self):
        if self.args.loss_weight_type == "numerical":
            return {"numeric_w": self.args.loss_numeric_w}
        elif self.args.loss_weight_type == "proportion":
            if self.args.loss_proportion_decay > 0:
                return {"proportion_decay": self.args.loss_proportion_decay}
        return {}

    def __call__(self, inputs, mode, *args, **kwargs):
        self._inputs = inputs
        self.mode = mode

        default_w_regu, default_b_regu = self._get_regularizer()
        default_w_init, default_b_init = self._get_initializer()

        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                            weights_regularizer=default_w_regu,
                            weights_initializer=default_w_init,
                            biases_regularizer=default_b_regu,
                            biases_initializer=default_b_init):
            with slim.arg_scope(self._net_arg_scope()):
                self._build_network(*args, **kwargs)

        if self.mode == ModeKeys.TRAIN:
            loss = self._build_loss()
            self._build_metrics()

            # Call _build_summaries() after _build_loss() to summarize losses and _build_metrics() to summarize metrics
            self._build_summaries()

            return loss
