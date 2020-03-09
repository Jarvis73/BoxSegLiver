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


class UNet(base.BaseNet):
    def __init__(self, args, name=None):
        """ Don't create ops/tensors in __init__() """
        super(UNet, self).__init__(args)
        self.name = name or "UNet"
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
                            biases_regularizer=default_b_regu,
                            outputs_collections=["EPts"]) as scope:
            if self.args.without_norm:
                return scope
            normalizer, params = self._get_normalization()
            with slim.arg_scope([slim.conv2d],
                                normalizer_fn=normalizer,
                                normalizer_params=params) as scope2:
                return scope2

    def _build_network(self, *args, **kwargs):
        out_channels = kwargs.get("init_channels", 64)
        num_down_samples = kwargs.get("num_down_samples", 4)

        # Tensorflow can not infer input tensor shape when constructing graph
        self.height = base._check_size_type(self.height)
        self.width = base._check_size_type(self.width)
        self._inputs["images"].set_shape([self.bs, self.height, self.width, self.channel])
        if "labels" in self._inputs:
            self._inputs["labels"].set_shape([self.bs, None, None])

        if self.args.img_grad:
            dy, dx = tf.image.image_gradients(self._inputs["images"])
            tensor_out = tf.concat((self._inputs["images"], dy, dx), axis=-1)
        else:
            tensor_out = self._inputs["images"]

        with tf.variable_scope(self.name, "UNet"):
            # encoder
            for i in range(num_down_samples):
                with tf.variable_scope("Encode{:d}".format(i + 1)):
                    tensor_out = slim.repeat(tensor_out, 2, slim.conv2d, out_channels, 3)   # Conv-BN-ReLU
                    self._layers["Encode{:d}".format(i + 1)] = tensor_out
                    tensor_out = slim.max_pool2d(tensor_out, [2, 2])
                out_channels *= 2

            # Encode-Decode-Bridge
            tensor_out = slim.repeat(tensor_out, 2, slim.conv2d, out_channels, 3, scope="ED-Bridge")

            # decoder
            for i in reversed(range(num_down_samples)):
                out_channels /= 2
                with tf.variable_scope("Decode{:d}".format(i + 1)):
                    tensor_out = slim.conv2d_transpose(tensor_out,
                                                       tensor_out.get_shape()[-1] // 2, 2, 2)
                    tensor_out = tf.concat((self._layers["Encode{:d}".format(i + 1)], tensor_out), axis=-1)
                    tensor_out = slim.repeat(tensor_out, 2, slim.conv2d, out_channels, 3)

            # final
            with slim.arg_scope([slim.conv2d],
                                activation_fn=None,
                                normalizer_fn=None, normalizer_params=None):
                logits = slim.conv2d(tensor_out, self.num_classes, 1, scope="AdjustChannels")
                self._layers["logits"] = logits

            # Probability & Prediction
            self.ret_prob = kwargs.get("ret_prob", False)
            self.ret_pred = kwargs.get("ret_pred", False)
            if self.ret_prob or self.ret_pred:
                self.probability = slim.softmax(logits)
                # if self.ret_prob:
                #     for i in range(1, self.num_classes):
                #         self.predictions[self.classes[i] + "Prob"] = split[i]
                if self.ret_pred:
                    split = tf.split(self.probability, self.num_classes, axis=-1)
                    zeros = tf.zeros_like(split[0], dtype=tf.uint8)
                    ones = tf.ones_like(zeros, dtype=tf.uint8)
                    for i in range(1, self.num_classes):
                        obj = self.classes[i] + "Pred"
                        self.predictions[obj] = tf.where(split[i] > 0.5, ones, zeros, name=obj)
                        self._image_summaries[obj] = self.predictions[obj]

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


if __name__ == "__main__":
    from tensorflow.contrib.layers.python.layers import utils
    from pprint import pprint
    from pathlib import Path

    class FOO(object):
        im_height = 256
        im_width = 256
        im_channel = 3
        batch_size = 1
        num_gpus = 1
        classes = ["Liver"]
        without_norm = False
        weight_decay_rate = 0.
        weight_init = "xavier"
        normalizer = "batch_norm"
        tag = "002_unet_liver"
    args = FOO()

    x = tf.placeholder(tf.float32, (1, 256, 256, 3), "Input")
    model = UNet(args)
    model({"images": x}, mode="infer", ret_prob=True)
    # pprint(tf.get_collection("EPts"))
    outputs = utils.convert_collection_to_dict("EPts")
    outputs = {k.replace("UNet/", "").replace("Repeat/", "").replace("/", "_").replace("convolution2d", "conv").
               replace("Conv2d_transpose", "up").replace("AdjustChannels", "out"):
               v for k, v in outputs.items()}
    outputs.update({"y": model.probability})
    saver = tf.train.Saver()
    with tf.Session() as sess:
        ckpt = tf.train.latest_checkpoint(Path(__file__).parent.parent / "model_dir" / args.tag,
                                          "checkpoint_best_125000")
        saver.restore(sess, ckpt)
        tf.saved_model.simple_save(sess, "export_dir/{}".format(args.tag), {"x": x}, outputs)
