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
from tensorflow.python.ops import init_ops

import loss_metrics as metrics

ModeKeys = tfes.estimator.ModeKeys


def _check_size_type(size):
    if size < 0:
        return None
    return size


class BaseNet(object):
    def __init__(self, args):
        # Private
        self._name = "Base"
        self._mode = ModeKeys.TRAIN
        self._args = args
        self._inputs = {}
        # self._layers saves some useful network layer outputs.
        # Keys that ends with `Pred` will be collected in model_fn for generating prediction dict
        self._layers = {}
        self._image_summaries = {}
        self.classes = ["Background"]
        self.metrics_dict = {}
        self.predictions = {}
        self.key_collections = {}

        self._is_training = self.mode == ModeKeys.TRAIN
        self._feed_dict = {}

        self.ret_prob = False
        self.ret_pred = False

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
            # self._is_training = self.mode == ModeKeys.TRAIN

            self._is_training = tf.placeholder_with_default(False, shape=(), name="is_training")
            self._feed_dict[self._is_training] = self.mode == ModeKeys.TRAIN
            tf.logging.info("Create graph in {} mode".format(new_mode))

    @property
    def is_training(self):
        return self._is_training

    @property
    def args(self):
        return self._args

    @property
    def num_classes(self):
        return len(self.classes)

    @property
    def layers(self):
        return self._layers

    @property
    def feed_dict(self):
        return self._feed_dict

    @property
    def metrics(self):
        all_metrics = {}
        for tensor in metrics.get_metrics():
            name = tensor.op.name
            parts = name.split("/")
            if parts[-1] == "value" and len(parts) > 1:
                all_metrics[parts[-2]] = tensor
            else:
                all_metrics[name] = tensor
        return all_metrics

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
        if self.args.weight_decay_rate > 0 and self.args.optimizer.lower() != "adamw":
            w_reg = slim.l2_regularizer(self.args.weight_decay_rate)
            b_reg = None if self.args.bias_decay else w_reg
        else:
            w_reg, b_reg = None, None

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

    def _get_normalization(self, freeze=None):
        if self.args.normalizer == "batch_norm":
            normalizer_params = {"scale": True}
            if freeze is None:
                normalizer_params.update({"is_training": self.is_training})
            elif not freeze:    # False
                normalizer_params.update({"is_training": True})
            else:
                normalizer_params.update({"is_training": False, "trainable": False})
            normalizer = slim.batch_norm
        elif self.args.normalizer == "instance_norm":
            normalizer_params = {}
            normalizer = slim.instance_norm
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

        with slim.arg_scope(self._net_arg_scope()):
            self._build_network(*args, **kwargs)

        ret = None
        if self.mode == ModeKeys.TRAIN:
            ret = self._build_loss()
        if kwargs.get("build_metrics", False):
            self._build_metrics()
        if kwargs.get("build_summaries", False):
            # Call _build_summaries() after _build_loss() to summarize losses and
            # _build_metrics() to summarize metrics
            self._build_summaries()

        return ret
