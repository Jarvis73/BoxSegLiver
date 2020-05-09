import tensorflow as tf
import tensorflow_estimator as tfes
import tensorflow.contrib.slim as slim

import loss_metrics as losses
from NetworksV2 import base
from NetworksV2.Backbone.vnet_backbone.activation_function import get_activation_fn
from NetworksV2.Backbone.vnet_backbone.layer import conv_bn_relu_drop, get_num_channels, concat_conv_bn_relu_drop

from utils import distribution_utils

ModeKeys = tfes.estimator.ModeKeys
metrics = losses


class _ModelConfig:
    config = {}
    config[0] = {
        'encoder_1': {'conv_block': {'kernel': [3, 3, 3], 'strides': [1, 1, 1], 'replace': 1, 'resnet': True},
                      'down_block': {'kernel': [3, 3, 3], 'strides': [2, 2, 2]}},
        'encoder_2': {'conv_block': {'kernel': [3, 3, 3], 'strides': [1, 1, 1], 'replace': 2, 'resnet': True},
                      'down_block': {'kernel': [3, 3, 3], 'strides': [2, 2, 2]}},
        'encoder_3': {'conv_block': {'kernel': [3, 3, 3], 'strides': [1, 1, 1], 'replace': 3, 'resnet': True},
                      'down_block': {'kernel': [3, 3, 3], 'strides': [2, 2, 2]}},
        'encoder_4': {'conv_block': {'kernel': [3, 3, 3], 'strides': [1, 1, 1], 'replace': 3, 'resnet': True},
                      'down_block': {'kernel': [3, 3, 3], 'strides': [2, 2, 2]}},
        'bottom': {'conv_block': {'kernel': [3, 3, 3], 'strides': [1, 1, 1], 'replace': 3, 'resnet': True}},
        'decoder_4': {'up_block': {'kernel': [3, 3, 3], 'strides': [1, 2, 2, 2, 1]},
                      'concat_conv_block': {'kernel': [3, 3, 3], 'strides': [1, 1, 1], 'replace': 3, 'resnet': True,
                                            'feature': 'encoder_4'}},
        'decoder_3': {'up_block': {'kernel': [3, 3, 3], 'strides': [1, 2, 2, 2, 1]},
                      'concat_conv_block': {'kernel': [3, 3, 3], 'strides': [1, 1, 1], 'replace': 3, 'resnet': True,
                                            'feature': 'encoder_3'}},
        'decoder_2': {'up_block': {'kernel': [3, 3, 3], 'strides': [1, 2, 2, 2, 1]},
                      'concat_conv_block': {'kernel': [3, 3, 3], 'strides': [1, 1, 1], 'replace': 3, 'resnet': True,
                                            'feature': 'encoder_2'}},
        'decoder_1': {'up_block': {'kernel': [3, 3, 3], 'strides': [1, 2, 2, 2, 1]},
                      'concat_conv_block': {'kernel': [3, 3, 3], 'strides': [1, 1, 1], 'replace': 3, 'resnet': True,
                                            'feature': 'encoder_1'}}
    }
    config[1] = {
        'encoder_1': {'conv_block': {'kernel': [1, 3, 3], 'strides': [1, 1, 1], 'replace': 1, 'resnet': True},
                      'down_block': {'kernel': [1, 3, 3], 'strides': [1, 2, 2]}},
        'encoder_2': {'conv_block': {'kernel': [1, 3, 3], 'strides': [1, 1, 1], 'replace': 2, 'resnet': True},
                      'down_block': {'kernel': [1, 3, 3], 'strides': [1, 2, 2]}},
        'encoder_3': {'conv_block': {'kernel': [1, 3, 3], 'strides': [1, 1, 1], 'replace': 3, 'resnet': True},
                      'down_block': {'kernel': [1, 3, 3], 'strides': [1, 2, 2]}},
        'encoder_4': {'conv_block': {'kernel': [3, 3, 3], 'strides': [1, 1, 1], 'replace': 3, 'resnet': True},
                      'down_block': {'kernel': [3, 3, 3], 'strides': [2, 2, 2]}},
        'bottom': {'conv_block': {'kernel': [3, 3, 3], 'strides': [1, 1, 1], 'replace': 3, 'resnet': True}},
        'decoder_4': {'up_block': {'kernel': [3, 3, 3], 'strides': [2, 2, 2]},
                      'concat_conv_block': {'kernel': [3, 3, 3], 'strides': [1, 1, 1], 'replace': 3, 'resnet': True,
                                            'feature': 'encoder_4'}},
        'decoder_3': {'up_block': {'kernel': [1, 3, 3], 'strides': [1, 2, 2]},
                      'concat_conv_block': {'kernel': [1, 3, 3], 'strides': [1, 1, 1], 'replace': 3, 'resnet': True,
                                            'feature': 'encoder_3'}},
        'decoder_2': {'up_block': {'kernel': [1, 3, 3], 'strides': [1, 2, 2]},
                      'concat_conv_block': {'kernel': [1, 3, 3], 'strides': [1, 1, 1], 'replace': 3, 'resnet': True,
                                            'feature': 'encoder_2'}},
        'decoder_1': {'up_block': {'kernel': [1, 3, 3], 'strides': [1, 2, 2]},
                      'concat_conv_block': {'kernel': [1, 3, 3], 'strides': [1, 1, 1], 'replace': 3, 'resnet': True,
                                            'feature': 'encoder_1'}}
    }


class VNet3D(base.BaseNet):
    def __init__(self, args, name=None):
        """ Don't create ops/tensors in __init__() """
        super(VNet3D, self).__init__(args)
        self.name = name or "VNet3D"
        self.classes.extend(self.args.classes)

        self.bs = distribution_utils.per_device_batch_size(args.batch_size, args.num_gpus)
        self.depth = args.im_depth
        self.height = args.im_height
        self.width = args.im_width
        self.channel = args.im_channel
        self.keep_prob = 1 - args.dropout

    def _net_arg_scope(self, *args, **kwargs):
        default_w_regu, default_b_regu = self._get_regularizer()
        default_w_init, _ = self._get_initializer()
        activation_fn = get_activation_fn('relu')
        with slim.arg_scope([slim.conv3d, slim.conv3d_transpose], weights_regularizer=default_w_regu,
                            weights_initializer=default_w_init, biases_regularizer=default_b_regu,
                            activation_fn=activation_fn):
            normalizer, params = self._get_normalization()
            with slim.arg_scope([slim.conv3d], normalizer_fn=normalizer, normalizer_params=params) as scope:
                return scope

    def _build_network(self, *args, **kwargs):
        self.depth = base._check_size_type(self.depth)
        self.height = base._check_size_type(self.height)
        self.width = base._check_size_type(self.width)
        # Tensorflow can not infer input tensor shape when constructing graph
        self._inputs["images"].set_shape([self.bs, self.depth, self.height, self.width, self.channel])

        init_channels = kwargs.get("init_channels", 32)
        num_pool_layers = kwargs.get("num_pool_layers", 1)
        activation_type = kwargs.get('activation_type', 'relu')
        norm_type = kwargs.get('norm_type', 'batch')

        net_configs = _ModelConfig.config[num_pool_layers]
        tf.logging.info("Model config: {}".format(kwargs))
        with tf.variable_scope(self.name):
            if hasattr(self.args, "img_grad") and self.args.img_grad:
                dz, dy, dx = tf.image.image_gradients(self._inputs["images"])
                inputs = tf.concat((self._inputs["images"], dz, dy, dx), axis=-1)
            else:
                inputs = self._inputs["images"]

            x = inputs
            with tf.variable_scope('input_layer'):
                x = conv_bn_relu_drop(x, init_channels, [3, 3, 3], [1, 1, 1], activation_type, norm_type,
                                      self.is_training, self.keep_prob, 1, None)
            features = {}
            for level_name, items in net_configs.items():
                with tf.variable_scope(level_name):
                    for sub_name, _configs in items.items():
                        n_channels = get_num_channels(x)
                        if 'conv_block' == sub_name:
                            x = conv_bn_relu_drop(x, n_channels, _configs['kernel'], _configs['strides'],
                                                  activation_type, norm_type, self.is_training, self.keep_prob,
                                                  _configs['replace'], _configs['resnet'])
                            features[level_name] = x
                        elif 'down_block' == sub_name:
                            x = slim.conv3d(x, n_channels * 2, kernel_size=_configs['kernel'],
                                            stride=_configs['strides'], scope='down_conv_bn_relu')
                        elif 'up_block' == sub_name:
                            x = slim.conv3d_transpose(x, n_channels // 2, kernel_size=_configs['kernel'],
                                                      stride=_configs['strides'], biases_initializer=None,
                                                      scope='deconv_bn_relu')
                        elif 'concat_conv_block' == sub_name:
                            feature = features[_configs['feature']]
                            x = concat_conv_bn_relu_drop(x, feature, n_channels, _configs['kernel'],
                                                         _configs['strides'], activation_type, norm_type,
                                                         self.is_training, self.keep_prob, _configs['replace'],
                                                         _configs['resnet'])
                        else:
                            raise Exception('找不到相应操作')
            logits = slim.conv3d(x, self.num_classes, kernel_size=1, activation_fn=None, scope='output_layer')
            self._layers["logits"] = logits

            # Probability & Prediction
            self.ret_prob = kwargs.get("ret_prob", True)
            self.ret_pred = kwargs.get("ret_pred", True)
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
            self._inputs["labels"].set_shape([self.bs, self.depth, self.height, self.width])
            w_param = self._get_weights_params()
            has_loss = False
            if "xentropy" in self.args.loss_type:
                losses.weighted_sparse_softmax_cross_entropy(logits=self._layers["logits"],
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
            with tf.name_scope("SumImage"):
                image = self._inputs["images"][:, self.depth // 2]
                images = [image - tf.reduce_min(image)]

            for i, image in enumerate(images):
                tf.summary.image("{}/Image{}".format(self.args.tag, i), image,
                                 max_outputs=1, collections=[self.DEFAULT])

            with tf.name_scope("SumLabel"):
                labels = tf.expand_dims(self._inputs["labels"][:, self.depth // 2], axis=-1)
                labels_uint8 = tf.cast(labels * 255 / len(self.args.classes), tf.uint8)
            tf.summary.image("{}/Label".format(self.args.tag), labels_uint8,
                             max_outputs=1, collections=[self.DEFAULT])

            for key, value in self._image_summaries.items():
                tf.summary.image("{}/{}".format(self.args.tag, key), value[:, self.depth // 2] * 255,
                                 max_outputs=1, collections=[self.DEFAULT])
