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
from tensorflow.contrib.layers.python.layers import utils as utils_lib

import loss_metrics as losses
from NetworksV2 import base
from NetworksV2.Backbone import slim_nets
from utils import distribution_utils

ModeKeys = tfes.estimator.ModeKeys
metrics = losses


def modulated_conv_block(self, net, repeat, channels, dilation=1, scope_id=0,
                         outputs_collections=None):
    with tf.variable_scope("down_conv{}".format(scope_id)):
        for i in range(repeat):
            with tf.variable_scope("mod_conv{}".format(i + 1)) as sc:
                net = slim.conv2d(net, channels, 3, rate=dilation)
                utils_lib.collect_named_outputs(outputs_collections, sc.name, net)
        return net


class UNetInter(base.BaseNet):
    def __init__(self, args, name=None):
        """ Don't create ops/tensors in __init__() """
        super(UNetInter, self).__init__(args)
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
        self.ct_conv = True if hasattr(args, "ct_conv") else False
        if self.ct_conv:
            tf.logging.info("Train: ////// Use conv context submodule")

    def _net_arg_scope(self, *args, **kwargs):
        default_w_regu, default_b_regu = self._get_regularizer()
        default_w_init, _ = self._get_initializer()

        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                            weights_regularizer=default_w_regu,
                            weights_initializer=default_w_init,
                            biases_regularizer=default_b_regu,
                            outputs_collections=["EPts"]):
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
        self._inputs["sp_guide"].set_shape([self.bs, self.height, self.width, 1])
        image = tf.concat((self._inputs["images"], self._inputs["sp_guide"]), axis=-1)

        base_channels = kwargs.get("init_channels", 64)
        num_down_samples = kwargs.get("num_down_samples", 4)
        tf.logging.info("Model config: {}".format(kwargs))

        with tf.variable_scope(self.name):
            def encoder_arg_scope():
                if self.args.without_norm:
                    return slim.current_arg_scope()
                else:
                    encoder_norm_params = {
                        'center': True,
                        'scale': True,
                    }
                    if self.args.normalizer == "batch_norm":
                        encoder_norm_params.update({
                            'decay': 0.99,
                            'is_training': self.is_training
                        })
                    with slim.arg_scope([slim.conv2d],
                                        normalizer_fn=self._get_normalization()[0],
                                        normalizer_params=encoder_norm_params,
                                        outputs_collections=None) as scope:
                        return scope

            # Encoder
            with tf.variable_scope("Encode"), slim.arg_scope(encoder_arg_scope()):
                nets = [image]
                for i in range(num_down_samples + 1):
                    net = modulated_conv_block(
                        self, nets[-1], 2, base_channels * 2 ** i, scope_id=i + 1,
                        outputs_collections=["EPts"])
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
            has_loss = False
            if "xentropy" in self.args.loss_type:
                losses.weighted_sparse_softmax_cross_entropy(logits=self._layers["logits"],
                                                             labels=self._inputs["labels"],
                                                             w_type=self.args.loss_weight_type, **w_param)
                has_loss = True
            if "dice" in self.args.loss_type:
                losses.weighted_dice_loss(logits=self.probability,
                                          labels=self._inputs["labels"],
                                          w_type=self.args.loss_weight_type, **w_param)
                has_loss = True
            if not has_loss:
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
                    ct = self._inputs["context"]
                    if not self.ct_conv:
                        image3 = tf.expand_dims(ct, axis=-1)
                        # Gaussian blur
                        kernel = tf.exp(-tf.convert_to_tensor([1., 0., 1.]) / (2 * 1.5 * 1.5)) / (2 * 3.14159 * 1.5 * 1.5)
                        kernel = tf.expand_dims(tf.expand_dims(kernel / tf.reduce_sum(kernel), axis=-1), axis=-1)
                        image3 = tf.nn.conv1d(image3, kernel, 1, padding="SAME")
                        image3 = tf.expand_dims(image3, axis=0)
                        images.append(tf.image.resize_nearest_neighbor(
                            image3, tf.concat(([self.bs * 10], tf.shape(self._inputs["context"])[1:]), axis=0),
                            align_corners=True))
                    else:
                        a, b, c = tf.split(ct, 3, axis=-1)
                        emp = tf.constant(1, shape=(self.bs, 32, 2, 1), dtype=ct.dtype) * \
                              tf.reduce_max(ct, axis=(1, 2, 3), keepdims=True)
                        image3 = tf.concat((a, emp, b, emp, c), axis=2, name="Context")
                        images.append(image3)

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

            for x in tf.global_variables():
                if "spatial" in x.op.name:
                    tf.summary.histogram(x.op.name.replace("GUNet/spatial", "sp"), x)


class FOO(object):
    im_height = 256
    im_width = 256
    im_channel = 3
    batch_size = 1
    num_gpus = 1
    classes = ["Liver", "Tumor"]
    without_norm = False
    weight_decay_rate = 0.
    weight_init = "xavier"
    normalizer = "instance_norm"
    tag = None
    use_spatial = True
    use_context = False
    side_dropout = True
    dropout = 0
    use_se = False
    enhance = False
    fix = False

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)


def dump_gunet_for_liver(tag, checkpoint="checkpoint_best", **kwargs):
    args = FOO(tag=tag, **kwargs)

    x = tf.placeholder(tf.float32, (args.batch_size, args.im_height, args.im_width, args.im_channel), "Input")
    # context = tf.placeholder(tf.float32, (1, 200), "Context")
    spatial = tf.placeholder(tf.float32, (args.batch_size, args.im_height, args.im_width, 1), "Spatial")
    model = GUNet(args)
    model({"images": x, "sp_guide": spatial}, mode="infer",
          init_channels=64,
          num_down_samples=4,
          mod_layers=[1, 2, 3, 4],
          context_fc_channels=[200, 200],
          context_model="fc",
          context_conv_init_channels=2,
          norm_with_center=False,
          norm_with_scale=True,
          ret_prob=False,
          ret_pred=True)
    outputs = utils.convert_collection_to_dict("EPts")
    rep_from = ["GUNet/", "Encode/", "Decode/", "Conv/", "/", "AdjustChannels"]
    rep_to = ["", "", "", "", "_", "out"]

    def apply_trans(x):
        for ft in zip(rep_from, rep_to):
            x = x.replace(*ft)
        return x

    outputs = {apply_trans(k): v for k, v in outputs.items()}
    outputs.update({"y": model.probability})
    pprint(outputs)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        ckpt = tf.train.latest_checkpoint(Path(__file__).parent.parent / "model_dir" / args.tag, checkpoint)
        saver.restore(sess, ckpt)
        tf.saved_model.simple_save(sess, "export_dir/{}".format(args.tag), {"x": x, "sp_guide": spatial}, outputs)


def dump_gunet_for_nf(tag, checkpoint="checkpoint_best", out_feat=False, **kwargs):
    args = FOO(tag=tag, im_height=960, im_width=320, classes=["NF"], **kwargs)
    x = tf.placeholder(tf.float32, (args.batch_size, None, None, args.im_channel), "Input")
    # context = tf.placeholder(tf.float32, (1, 200), "Context")
    spatial = tf.placeholder(tf.float32, (args.batch_size, None, None, 1), "Spatial")
    model = GUNet(args)
    model({"images": x, "sp_guide": spatial}, mode="infer",
          init_channels=64,
          num_down_samples=4,
          mod_layers=[1, 2, 3, 4],
          context_fc_channels=[200, 200],
          context_model="fc",
          context_conv_init_channels=2,
          norm_with_center=False,
          norm_with_scale=True,
          ret_prob=False,
          ret_pred=True)
    if out_feat:
        outputs = utils.convert_collection_to_dict("EPts")
        rep_from = ["GUNet/", "Encode/", "Decode/", "Conv/", "/", "AdjustChannels"]
        rep_to = ["", "", "", "", "_", "out"]

        def apply_trans(x):
            for ft in zip(rep_from, rep_to):
                x = x.replace(*ft)
            return x

        outputs = {apply_trans(k): v for k, v in outputs.items()}
        outputs.update({"y": model.probability})
        pprint(outputs)
    else:
        outputs = {"y_prob": model.probability, "y_pred": model.predictions["NFPred"]}

    saver = tf.train.Saver()
    with tf.Session() as sess:
        ckpt = tf.train.latest_checkpoint(Path(__file__).parent.parent / "model_dir" / args.tag, checkpoint)
        saver.restore(sess, ckpt)
        tf.saved_model.simple_save(sess, "export_dir/{}".format(args.tag), {"x": x, "sp_guide": spatial}, outputs)


def export_model(tag, export_model_dir, model_version=1, checkpoint="checkpoint_best", **kwargs):
    args = FOO(tag=tag, im_height=-1, im_width=-1, **kwargs)

    with tf.Session(graph=tf.Graph()) as sess:
        image_height_tensor = tf.placeholder(tf.int32)
        image_width_tensor = tf.placeholder(tf.int32)
        serialized_tf_example = tf.placeholder(tf.string, name="tf_example")
        feature_configs = {"x": tf.FixedLenFeature(shape=[], dtype=tf.float32),
                           "g": tf.FixedLenFeature(shape=[], dtype=tf.float32)}
        tf_example = tf.parse_example(serialized_tf_example, feature_configs)

        tf_example["x"] = tf.reshape(tf_example["x"], (1, image_height_tensor, image_width_tensor, 3))
        tf_example["g"] = tf.reshape(tf_example["g"], (1, image_height_tensor, image_width_tensor, 1))
        image = tf.identity(tf_example["x"], name="image")
        guide = tf.identity(tf_example["g"], name="guide")

        model = GUNet(args)
        model({"images": image, "sp_guide": guide}, mode="infer",
              init_channels=64,
              num_down_samples=4,
              mod_layers=[1, 2, 3, 4],
              context_fc_channels=[200, 200],
              context_model="fc",
              context_conv_init_channels=2,
              norm_with_center=False,
              norm_with_scale=True,
              ret_prob=False,
              ret_pred=False)
        logits_tf = model._layers["logits"]
        softmax = tf.nn.softmax(logits_tf)
        predictions_tf = tf.argmax(softmax, axis=3)

        saver = tf.train.Saver()
        ckpt = tf.train.latest_checkpoint(Path(__file__).parent.parent / "model_dir" / args.tag, checkpoint)

        saver.restore(sess, ckpt)
        print("Model", tag, "restored.")

        export_path = Path(export_model_dir) / tag / str(model_version)
        print('Exporting trained model to', str(export_path))
        builder = tf.saved_model.builder.SavedModelBuilder(str(export_path))

        tensor_info_image = tf.saved_model.utils.build_tensor_info(image)
        tensor_info_guide = tf.saved_model.utils.build_tensor_info(guide)
        tensor_info_height = tf.saved_model.utils.build_tensor_info(image_height_tensor)
        tensor_info_width = tf.saved_model.utils.build_tensor_info(image_width_tensor)
        tensor_info_output = tf.saved_model.utils.build_tensor_info(predictions_tf)
        # tensor_info_output2 = tf.saved_model.utils.build_tensor_info(softmax[..., 1])
        # tensor_info_output3 = tf.saved_model.utils.build_tensor_info(logits_tf)

        # make sure keys ends with '_bytes' for images
        prediction_signature = (
            tf.saved_model.signature_def_utils.build_signature_def(
                inputs={'images': tensor_info_image, 'guide': tensor_info_guide,
                        'height': tensor_info_height, 'width': tensor_info_width},
                outputs={'out_pred': tensor_info_output,
                         # 'out_prob': tensor_info_output2,
                         # "out_logit": tensor_info_output3
                         },
                method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))

        builder.add_meta_graph_and_variables(
            sess, [tf.saved_model.tag_constants.SERVING],
            signature_def_map={
                'serving_default':
                    prediction_signature,
            })

        builder.save(as_text=True)
        print('Done exporting!')


if __name__ == "__main__":
    from tensorflow.contrib.layers.python.layers import utils
    from pprint import pprint
    from pathlib import Path

    # dump_gunet_for_nf("111_nf_sp_rand")
    export_model("112_nf_sp_fix", "export_path", model_version=1, checkpoint="checkpoint",
                 classes=["NF"])
