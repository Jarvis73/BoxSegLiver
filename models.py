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

import copy
import yaml  # conda install -c conda-forge pyyaml
import tensorflow as tf
from pathlib import Path
from tensorflow.python import pywrap_tensorflow as pt

from Networks.UNet import UNet
from Networks.OSMN import OSMNUNet

ModeKeys = tf.estimator.ModeKeys

# Available models
MODEL_ZOO = [
    UNet,
    OSMNUNet
]


def add_arguments(parser):
    group = parser.add_argument_group(title="Model Arguments")
    group.add_argument("--model",
                       type=str,
                       choices=[cls.__name__ for cls in MODEL_ZOO],
                       required=True, help="Model backbone")
    group.add_argument("--model_config",
                       type=str,
                       required=False, help="Model configuration. (default: <model>.yml)")
    group.add_argument("--classes",
                       type=str,
                       nargs="+",
                       required=True, help="Class names of the objects")
    group.add_argument("--batch_size",
                       type=int,
                       default=8,
                       required=False, help="Model batch size (default: %(default)d)")
    group.add_argument("--weight_init",
                       type=str,
                       default="xavier",
                       choices=["trunc_norm", "xavier"],
                       required=False, help="Model variable initialization method (default: %(default)s)")
    group.add_argument("--normalizer",
                       type=str,
                       default="batch_norm",
                       choices=["batch_norm"],
                       required=False, help="Normalization method (default: %(default)s)")
    group.add_argument("--cls_branch",
                       action="store_true",
                       required=False, help="Classify branch")
    group.add_argument("--load_weights",
                       type=str,
                       required=False, help="Initialize some of the model parameters from this given "
                                            "ckpt file.")
    group.add_argument("--weights_scope",
                       type=str,
                       required=False, help="Network scope of the weights in the given ckpt file, which "
                                            "will be replaced with current network scope. If not provide,"
                                            " it will be inferred")
    group.add_argument("--without_bn",
                       action="store_true",
                       required=False, help="Conv without batch normalization")


def get_model_params(args):
    params = {}

    if False:   # Add sophisticated models
        pass
    else:   # Simpler model (only need "args" to initialize)
        params["model"] = eval(args.model)
        if not args.model_config:
            args.model_config = args.model + ".yml"
        model_config_path = Path(__file__).parent / "Networks" / args.model_config
        with model_config_path.open() as f:
            params["model_kwargs"] = yaml.load(f)

    return params


def _find_root_scope(ckpt_filename):
    reader = pt.NewCheckpointReader(ckpt_filename)
    variables = list(reader.get_variable_to_shape_map())
    for var in variables:
        if var.startswith("Optimizer") and not var.endswith("power"):
            return var.split("/")[1]
        continue


def init_partial_model(model, args):
    if not args.load_weights:
        return None

    weights_dir = Path(args.model_dir).parent / args.load_weights
    ckpt_filename = args.load_weights
    if weights_dir.is_dir():
        ckpt = tf.train.get_checkpoint_state(str(weights_dir), latest_filename="checkpoint_best")
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_filename = ckpt.model_checkpoint_path

    if not tf.train.checkpoint_exists(ckpt_filename):
        raise FileNotFoundError("ckpt_filename {} doesn't exist".format(ckpt_filename))

    root_scope = args.weights_scope or _find_root_scope(ckpt_filename)
    model_vars = tf.global_variables(".*?{}/(?!BboxLayer)".format(model.name))
    var_list = {var.op.name.replace(model.name, root_scope): var for var in model_vars}
    # This is an one-off Saver. Do not add it to GraphKeys.SAVERS collection.
    saver = tf.train.Saver(var_list=var_list)
    tf.logging.info("Create init_fn with checkpoint: " + ckpt_filename)

    def init_fn(scaffold, session):
        _ = scaffold
        saver.restore(session, ckpt_filename)

    return init_fn


def model_fn(features, labels, mode, params):
    features = copy.copy(features)
    # Add graph nodes for images and labels
    images = tf.identity(features.pop("images"), name="Images")
    if labels is not None:
        labels = tf.identity(labels, name="Labels")
    elif "labels" in features:
        labels = tf.identity(features.pop("labels"), name="Labels")

    inputs = {"images": images, "labels": labels}
    inputs.update(features)

    train_op = None
    predictions = None

    args = params["args"]
    #############################################################################
    # create model
    model = params["model"](args)
    if "model_instances" not in params:
        params["model_instances"] = []
    params["model_instances"].append(model)
    model_args = params.get("model_args", ())
    model_kwargs = params.get("model_kwargs", {})

    loss = model(inputs, mode, *model_args, **model_kwargs)

    if mode == ModeKeys.TRAIN:
        # create solver
        solver = params["solver"]
        solver_args = params.get("solver_args", ())
        solver_kwargs = params.get("solver_kwargs", {})
        train_op = solver(loss, *solver_args, **solver_kwargs)

    if not args.train_without_eval or mode == ModeKeys.PREDICT:
        predictions = {key: value for key, value in model.layers.items()
                       if key.endswith("Pred")}
        predictions.update(model.metrics_dict)

        with tf.name_scope("LabelProcess/"):
            one_hot_label = tf.one_hot(labels, model.num_classes)
            split_labels = tf.split(one_hot_label, model.num_classes, axis=-1)

        for i, split_label in enumerate(split_labels[1:]):
            predictions["Labels_{}".format(i)] = split_label

        if args.resize_for_batch:
            predictions["Bboxes"] = features["bboxes"]

        predictions.update({
            "Names": features["names"],
            "Pads": features["pads"]
        })

    if mode == ModeKeys.TRAIN:
        predictions["GlobalStep"] = tf.train.get_global_step(tf.get_default_graph())
    if "livers" in features:
        predictions["BgMasks"] = features["livers"]
    if mode == ModeKeys.PREDICT and not args.eval_3d:
        predictions["Indices"] = features["indices"]

    #############################################################################
    kwargs = {"loss": loss,
              "train_op": train_op,
              "predictions": predictions}

    #############################################################################
    # Initialize partial graph variables by init_fn
    init_fn = init_partial_model(model, args)
    kwargs["scaffold"] = tf.train.Scaffold(init_fn=init_fn)

    return tf.estimator.EstimatorSpec(mode=mode, **kwargs)
