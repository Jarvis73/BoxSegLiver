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

ModeKeys = tf.estimator.ModeKeys
metrics = losses


def _check_size_type(size):
    if size < 0:
        return None
    return size


class DiscrimUNet(base.BaseNet):
    """ Only for tumor segmentation (without liver) """
    def __init__(self, args, name=None):
        """ Don't create ops/tensors in __init__() """
        super(DiscrimUNet, self).__init__(args)
        self.name = name or "DiscrimUNet"
        self.classes.extend(self.args.classes)

        self.bs = args.batch_size
        self.roi_bs = args.roi_batch_size
        self.height = args.im_height
        self.width = args.im_width
        self.channel = args.im_channel

    def _net_arg_scope(self, *args, **kwargs):
        default_w_regu, default_b_regu = self._get_regularizer()
        default_w_init, _ = self._get_initializer()
        normalizer, params = self._get_normalization()

        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                            weights_regularizer=default_w_regu,
                            weights_initializer=default_w_init,
                            biases_regularizer=default_b_regu):
            with slim.arg_scope([slim.conv2d],
                                normalizer_fn=normalizer,
                                normalizer_params=params) as scope:
                return scope

    def _build_network(self, *args, **kwargs):
        out_channels = kwargs.get("init_channels", 64)
        num_down_samples = kwargs.get("num_down_samples", 4)

        # Tensorflow can not infer input tensor shape when constructing graph
        self.height = _check_size_type(self.height)
        self.width = _check_size_type(self.width)
        self._inputs["images"].set_shape([self.bs, self.height, self.width, self.channel])
        self._inputs["labels"].set_shape([self.bs, None, None])
        self._inputs["livers"].set_shape([self.bs, self.height, self.width])
        self._inputs["rois"].set_shape([self.roi_bs, 5])
        self._inputs["roi_labels"].set_shape([self.roi_bs])

        tensor_out = self._inputs["images"]
        down_sample = kwargs.get("input_down_sample", False)
        h, w = self._inputs["images"].shape[1:3]
        if down_sample:
            rate = kwargs.get("input_down_sample_rate", 2)
            tensor_out = tf.image.resize_bilinear(tensor_out, [h // rate, w // rate])

        with tf.variable_scope(self.name, "DiscrimUNet"):
            encoder_layers = {}

            # encoder
            for i in range(num_down_samples):
                with tf.variable_scope("Encode{:d}".format(i + 1)):
                    tensor_out = slim.repeat(tensor_out, 2, slim.conv2d, out_channels, 3)   # Conv-BN-ReLU
                    encoder_layers["Encode{:d}".format(i + 1)] = tensor_out
                    tensor_out = slim.max_pool2d(tensor_out, [2, 2])
                out_channels *= 2

            # Encode-Decode-Bridge
            high_level_tensor_out = slim.repeat(tensor_out, 2, slim.conv2d, out_channels, 3, scope="ED-Bridge")

            # Branch 1: decoder for segmentation
            for i in reversed(range(num_down_samples)):
                out_channels /= 2
                with tf.variable_scope("Decode{:d}".format(i + 1)):
                    tensor_out = slim.conv2d_transpose(high_level_tensor_out,
                                                       tensor_out.get_shape()[-1] // 2, 2, 2)
                    tensor_out = tf.concat((encoder_layers["Encode{:d}".format(i + 1)], tensor_out), axis=-1)
                    tensor_out = slim.repeat(tensor_out, 2, slim.conv2d, out_channels, 3)

            # final
            with slim.arg_scope([slim.conv2d],
                                activation_fn=None,
                                normalizer_fn=None, normalizer_params=None):
                logits = slim.conv2d(tensor_out, self.num_classes, 1, scope="AdjustChannels")
                if down_sample:
                    logits = tf.image.resize_bilinear(logits, [h, w])
                self._layers["logits"] = logits

            # Branch 2: discriminator for classification
            with tf.name_scope("BboxLayer"):
                # bool_split = tf.squeeze(tf.cast(self._layers[obj], tf.bool), axis=-1)
                # labeled = image_ops.connected_components(bool_split)
                pooling_size = kwargs.get("pooling_size", 8)
                pre_pooling_size = pooling_size * 2
                rois = tf.image.crop_and_resize(image=high_level_tensor_out,
                                                bboxes=self._inputs["rois"][:, 1:],
                                                box_ind=self._inputs["rois"][:, 0],
                                                crop_size=[pre_pooling_size, pre_pooling_size])
                rois = slim.max_pool2d(rois, 2)
                rois_flat = slim.flatten(rois)
                fc = slim.fully_connected(rois_flat, 1024)
                fc = slim.dropout(fc, keep_prob=0.5, is_training=self.is_training)
                fc = slim.fully_connected(fc, 1024)
                fc = slim.dropout(fc, keep_prob=0.5, is_training=self.is_training)
                cls_score = slim.fully_connected(fc, 2, activation_fn=None)
                self._layers["Classify"] = cls_score

            with tf.name_scope("Prediction"):
                # Segmentation
                self.probability = slim.softmax(logits)
                split = tf.split(self.probability, self.num_classes, axis=-1)
                if self.ret_prob:
                    for i in range(1, self.num_classes):
                        self._layers[self.classes[i] + "Prob"] = split[i]

                zeros = tf.zeros_like(split[0], dtype=tf.uint8)
                ones = tf.ones_like(zeros, dtype=tf.uint8)
                for i in range(1, self.num_classes):
                    obj = self.classes[i] + "Pred"
                    self._layers[obj] = tf.where(split[i] > 0.5, ones, zeros, name=obj)
                    if self.args.only_tumor and self.classes[i] == "Tumor":
                        self._layers[obj] = self._layers[obj] * tf.cast(
                            tf.expand_dims(self._inputs["livers"], axis=-1), tf.uint8)
                    self._image_summaries[obj] = self._layers[obj]

                # Classification
                cls_prob = slim.softmax(cls_score)
                self._layers["ClassifyPred"] = tf.cast(tf.argmax(cls_prob, axis=1),
                                                       self._inputs["roi_labels"].dtype)
        return

    def _build_loss(self):
        # Segmentation loss
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

        # Classification loss
        cls_params = {
            "logits": self._layers["Classify"],
            "labels": self._inputs["roi_labels"],
            "scope": "ClsLoss"
        }
        tf.losses.sparse_softmax_cross_entropy(**cls_params)

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

                # Classification
                is_correct = tf.to_float(tf.equal(self._layers["ClassifyPred"],
                                                  self._inputs["roi_labels"]))
                accuracy = tf.reduce_mean(is_correct, name="Accuracy")
                tf.add_to_collection(metrics.METRICS, accuracy)

    def _build_summaries(self):
        if self.mode == ModeKeys.TRAIN:
            # Make sure all the elements are positive
            images = []
            if self.args.im_channel == 1:
                images.append(self._inputs["images"])
            elif self.args.im_channel == 2:
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
            labels_uint8 = tf.cast(labels * 255 / len(self.args.classes), tf.uint8)
            if self.args.only_tumor:
                livers_uint8 = tf.cast(tf.expand_dims(self._inputs["livers"], axis=-1)
                                       * 255 / len(self.args.classes), tf.uint8)
                tf.summary.image("{}/Liver".format(self.args.tag), livers_uint8,
                                 max_outputs=1, collections=[self.DEFAULT])
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
