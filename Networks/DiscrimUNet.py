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
from Networks.layers.generate_anchors import generate_anchors_pre
from Networks.layers.anchor_target_layer import anchor_target_layer

ModeKeys = tf.estimator.ModeKeys
metrics = losses


def _check_size_type(size):
    if size < 0:
        return None
    return size


def debug(tensor):
    ops = tf.print(tensor.op.name, tf.shape(tensor))
    with tf.control_dependencies([ops]):
        return tf.identity(tensor)


class DiscrimUNet(base.BaseNet):
    """ Only for tumor segmentation (without liver) """
    def __init__(self, args, name=None):
        """ Don't create ops/tensors in __init__() """
        super(DiscrimUNet, self).__init__(args)
        self.name = name or "DiscrimUNet"
        self.classes.extend(self.args.classes)

        self.bs = args.batch_size
        self.height = args.im_height
        self.width = args.im_width
        self.channel = args.im_channel
        self.metrics_dict = {}

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

    def _anchor_component(self):
        """ Wrapper function for generate_anchors_pre """
        with tf.variable_scope("Anchors"):
            height = tf.to_int32(tf.to_float(self.height) * 1.0 / self.feature_stride)
            width = tf.to_int32(tf.to_float(self.width) * 1.0 / self.feature_stride)

            # [w * h * A, 4]
            anchors, anchor_length = tf.py_func(generate_anchors_pre,
                                                [height, width, self.feature_stride,
                                                 self.anchor_scales, self.anchor_ratios],
                                                [tf.float32, tf.int32],
                                                name="gen_anchors")
            anchors.set_shape([None, 4])
            anchor_length.set_shape([])
            self._anchors = anchors
            self._anchor_length = anchor_length

    def _anchor_target_layer(self, cls_score):
        """ Wrapper function for anchor_target_layer """
        cls_labels = tf.py_func(anchor_target_layer,
                                [cls_score, self._inputs["gt_boxes"], self._inputs["im_info"],
                                 self._anchors, self._num_anchors],
                                tf.float32)
        cls_labels.set_shape([1, 1, None, None])
        self.cls_labels = tf.to_int32(cls_labels)

    @staticmethod
    def _reshape_layer(bottom, num_dim, name):
        input_shape = tf.shape(bottom)
        with tf.variable_scope(name):
            # change the channel to the caffe format
            to_caffe = tf.transpose(bottom, [0, 3, 1, 2])
            # then force it to have channel 2
            reshaped = tf.reshape(to_caffe,
                                  tf.concat(axis=0, values=[[1, num_dim, -1], [input_shape[2]]]))
            # then swap the channel back
            to_tf = tf.transpose(reshaped, [0, 2, 3, 1])
            return to_tf

    def _build_network(self, *args, **kwargs):
        out_channels = kwargs.get("init_channels", 64)
        num_down_samples = kwargs.get("num_down_samples", 4)
        self.feature_stride = 2 ** num_down_samples
        self.anchor_scales = kwargs.get("anchor_scales", [0, 1, 2])
        self.anchor_ratios = kwargs.get("anchor_ratios", [1])
        self._num_anchors = len(self.anchor_ratios) * len(self.anchor_scales)

        # Tensorflow can not infer input tensor shape when constructing graph
        self.height = _check_size_type(self.height)
        self.width = _check_size_type(self.width)
        self._inputs["images"].set_shape([self.bs, self.height, self.width, self.channel])
        self._inputs["labels"].set_shape([self.bs, None, None])
        self._inputs["gt_boxes"].set_shape([self.bs, None, 4])
        self._inputs["im_info"].set_shape([self.bs, 2])
        # self.bs == 1
        self._inputs["gt_boxes"] = tf.squeeze(self._inputs["gt_boxes"], axis=0)
        self._inputs["im_info"] = tf.squeeze(self._inputs["im_info"], axis=0)
        self.height, self.width = tf.split(self._inputs["im_info"], 2)

        tensor_out = self._inputs["images"]

        with tf.variable_scope(self.name, "DiscrimUNet"):
            encoder_layers = {}

            # image to head
            for i in range(num_down_samples):
                with tf.variable_scope("Encode{:d}".format(i + 1)):
                    tensor_out = slim.repeat(tensor_out, 2, slim.conv2d, out_channels, 3)   # Conv-BN-ReLU
                    encoder_layers["Encode{:d}".format(i + 1)] = tensor_out
                    tensor_out = slim.max_pool2d(tensor_out, [2, 2])
                out_channels *= 2
            high_level_tensor_out = slim.repeat(tensor_out, 2, slim.conv2d, out_channels, 3, scope="ED-Bridge")

            # Branch 1: decode for segmentation
            tensor_out = high_level_tensor_out
            for i in reversed(range(num_down_samples)):
                out_channels /= 2
                with tf.variable_scope("Decode{:d}".format(i + 1)):
                    tensor_out = slim.conv2d_transpose(tensor_out,
                                                       tensor_out.get_shape()[-1] // 2, 2, 2)
                    tensor_out = tf.concat((encoder_layers["Encode{:d}".format(i + 1)], tensor_out), axis=-1)
                    tensor_out = slim.repeat(tensor_out, 2, slim.conv2d, out_channels, 3)
            with slim.arg_scope([slim.conv2d],
                                activation_fn=None,
                                normalizer_fn=None, normalizer_params=None):
                logits = slim.conv2d(tensor_out, self.num_classes, 1, scope="AdjustChannels")
                self._layers["logits"] = logits

            # Branch 2: discriminator for classification
            with tf.name_scope("BboxLayer"):
                self._anchor_component()
                cls_layer = slim.conv2d(high_level_tensor_out,
                                        kwargs.get("cls_layer_channels", 512),
                                        kernel_size=3,
                                        scope="cls_conv-3x3")
                cls_score = slim.conv2d(cls_layer, self._num_anchors * 2, 1,
                                        padding="VALID",
                                        activation_fn=None,
                                        scope="cls_score")
                cls_score_reshape = self._reshape_layer(cls_score, 2, "cls_score_reshape")
                cls_prob_reshape = slim.softmax(cls_score_reshape, "cls_prob_reshape")
                cls_prob = self._reshape_layer(cls_prob_reshape, self._num_anchors * 2, "cls_prob")
                self._layers["cls_score_reshape"] = cls_score_reshape
                self._layers["cls_prob"] = cls_prob
                self._anchor_target_layer(cls_score)

            with tf.name_scope("Prediction"):
                self.ret_prob = kwargs.get("ret_prob", False)
                self.ret_pred = kwargs.get("ret_pred", False)
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
                    # if self.args.only_tumor and self.classes[i] == "Tumor":
                    #     self._layers[obj] = self._layers[obj] * tf.cast(
                    #         tf.expand_dims(self._inputs["livers"], axis=-1), tf.uint8)
                    self._image_summaries[obj] = self._layers[obj]

            with tf.name_scope("ClsPrediction"):
                # Classification
                cls_pred = tf.argmax(tf.reshape(cls_score_reshape, [-1, 2]), axis=1,
                                     name="cls_pred", output_type=tf.int32)
                self._layers["ClsPred"] = cls_pred
        return

    def _build_loss(self):
        with tf.name_scope("Losses/"):
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
            cls_score = tf.reshape(self._layers["cls_score_reshape"], [-1, 2])
            label = tf.reshape(self.cls_labels, [-1])
            self.select = tf.where(tf.not_equal(label, -1))
            cls_score = tf.reshape(tf.gather(cls_score, self.select), [-1, 2])
            self.label = tf.reshape(tf.gather(label, self.select), [-1])
            cls_loss = losses.sparse_focal_loss(cls_score, self.label)
            tf.losses.add_loss(cls_loss)
            # tf.losses.sparse_softmax_cross_entropy(logits=cls_score, labels=self.label, scope="ClsLoss")

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

                # Classification
                cls_pred = tf.reshape(tf.gather(self._layers["ClsPred"], self.select), [-1])
                is_correct = tf.to_float(tf.equal(cls_pred, self.label))
                accuracy = tf.reduce_mean(is_correct, name="Accuracy/value")
                self.metrics_dict["ClsAcc"] = accuracy
                tf.add_to_collection(metrics.METRICS, accuracy)

                positive = tf.to_float(tf.reduce_sum(self.label), name="Pos/value")
                pre_b = tf.cast(cls_pred, tf.bool)
                lab_b = tf.cast(self.label, tf.bool)
                tp = tf.reduce_sum(tf.to_float(tf.logical_and(pre_b, lab_b)))
                recall = tf.div(tp + 1e-6,  positive + 1e-6, name="Recall/value")
                self.metrics_dict["ClsPos"] = positive
                self.metrics_dict["ClsRec"] = recall
                tf.add_to_collection(metrics.METRICS, positive)
                tf.add_to_collection(metrics.METRICS, recall)

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
