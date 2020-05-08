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
import tensorflow_estimator as tfes
from pathlib import Path
from tensorflow.python import pywrap_tensorflow as pt

from NetworksV2.UNet import UNet
from NetworksV2.GUNet import GUNet
from NetworksV2.UNetInter import UNetInter
from NetworksV2.LGNet import LGNet
from NetworksV2.UNet3D import UNet3D
from NetworksV2.SmallUNet import SmallUNet
from NetworksV2.InterUNet import InterUNet
# from NetworksV2.DenseUNet import DenseUNet
from entry import infer_2d

ModeKeys = tfes.estimator.ModeKeys
# Available models
MODEL_ZOO = [
    UNet, GUNet, UNetInter, LGNet, UNet3D, SmallUNet, InterUNet  # , DenseUNet
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
                       choices=["batch_norm", "instance_norm"],
                       required=False, help="Normalization method (default: %(default)s)")
    group.add_argument("--cls_branch",
                       action="store_true",
                       required=False, help="Classify branch")
    group.add_argument("--load_weights",
                       type=str,
                       required=False, help="Initialize model parameters from this given ckpt file.")
    group.add_argument("--load_weights_version",
                       type=str,
                       default="checkpoint", help="Used for latest_filename")
    group.add_argument("--weights_scope",
                       type=str,
                       required=False, help="Network scope of the weights in the given ckpt file, which "
                                            "will be replaced with current network scope. If not provide,"
                                            " it will be inferred")
    group.add_argument("--without_norm",
                       action="store_true",
                       required=False, help="Conv without batch normalization")
    group.add_argument("--batches_per_epoch", type=int, default=2000, help="Number of batches per epoch")
    group.add_argument("--eval_per_epoch", action="store_true")
    group.add_argument("--dropout", type=float, help="Dropout for backbone networks")
    group.add_argument("--img_grad", action="store_true", help="Use image gradients")
    group.add_argument("--mid_cat", action="store_true", help="Concat guide to middle layers")


def get_model_params(args, build_metrics=False, build_summaries=False):
    params = dict()
    params["model"] = eval(args.model)
    tf.logging.info("Use {} for learning.".format(args.model))

    if not args.model_config:
        args.model_config = args.model + ".yml"

    # Try to find model config file in NetworksV2/ and NetworksV2/ext_config/ directories
    model_config_path = Path(__file__).parent.parent / "NetworksV2" / args.model_config
    if not model_config_path.exists():
        model_config_path = model_config_path.parent / "ext_config" / args.model_config
        if not model_config_path.exists():
            tf.logging.info("Cannot find model config file %s" % args.model_config)
            model_config_path = None

    if model_config_path:
        with model_config_path.open() as f:
            params["model_kwargs"] = yaml.load(f, Loader=yaml.Loader)
            tf.logging.info("Load model configuration from %s" % str(model_config_path))
    else:
        params["model_kwargs"] = {}

    params["model_kwargs"]["build_metrics"] = build_metrics
    params["model_kwargs"]["build_summaries"] = build_summaries

    return params


def get_model_2d_params(args):
    params = dict()
    params["model"] = eval(args.model_2d)

    tf.logging.info("Use {} for learning.".format(args.model))

    if not args.model_2d_config:
        args.model_2d_config = args.model_2d + ".yml"

    # Try to find model config file in NetworksV2/ and NetworksV2/ext_config/ directories
    model_config_path = Path(__file__).parent.parent / "NetworksV2" / args.model_2d_config
    if not model_config_path.exists():
        model_config_path = model_config_path.parent / "ext_config" / args.model_2d_config
        if not model_config_path.exists():
            tf.logging.info("Cannot find model config file %s" % args.model_config)
            model_config_path = None

    if model_config_path:
        with model_config_path.open() as f:
            params["model_kwargs"] = yaml.load(f, Loader=yaml.Loader)
            tf.logging.info("Load model configuration from %s" % str(model_config_path))
    else:
        params["model_kwargs"] = {}

    params["model_kwargs"]["build_metrics"] = False
    params["model_kwargs"]["build_summaries"] = False

    return params


def _find_root_scope(ckpt_filename):
    reader = pt.NewCheckpointReader(ckpt_filename)
    variables = list(reader.get_variable_to_shape_map())
    for var in variables:
        if var.startswith("Optimizer") and not var.endswith("power"):
            return var.split("/")[1]
        continue


def init_model(model, args):
    if not args.load_weights:
        return None

    weights_dir = Path(args.model_dir).parent / args.load_weights
    ckpt_filename = args.load_weights
    if weights_dir.is_dir():
        ckpt = tf.train.get_checkpoint_state(str(weights_dir), latest_filename=args.load_weights_version)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_filename = ckpt.model_checkpoint_path

    if not tf.train.checkpoint_exists(ckpt_filename):
        raise FileNotFoundError("ckpt_filename {} doesn't exist".format(ckpt_filename))

    root_scope = args.weights_scope or _find_root_scope(ckpt_filename)
    model_vars = tf.global_variables(model.name)
    var_list = {var.op.name.replace(model.name, root_scope): var for var in model_vars}
    # This is an one-off Saver. Do not add it to GraphKeys.SAVERS collection.
    saver = tf.train.Saver(var_list=var_list)
    tf.logging.info("Create init_fn with checkpoint: " + ckpt_filename)

    def init_fn(scaffold, session):
        _ = scaffold
        saver.restore(session, ckpt_filename)

    return init_fn


def init_dense_model():
    import h5py
    f = h5py.File(str(Path(__file__).parent.parent / "densenet161_weights_tf.h5"), "r")
    weight_keys = list(f.keys())
    all_assign = []
    variables = tf.trainable_variables("DenseUNet")
    for variable in variables:
        names = variable.op.name.split("/")
        layer = names[1]
        if layer in weight_keys:
            var = names[2]
            if var == "weights":
                value = f[layer][layer + "_W"].value
            elif var == "moving_mean":
                value = f[layer][layer + "_running_mean"].value
            elif var == "moving_variance":
                value = f[layer][layer + "_running_std"].value
            elif var == "beta":
                layer = layer[:-3] + "_scale"
                value = f[layer][layer + "_beta"].value
            elif var == "gamma":
                layer = layer[:-3] + "_scale"
                value = f[layer][layer + "_gamma"].value
            else:
                raise ValueError(variable, "missing data in weights file.")
            all_assign.append(tf.assign(variable, value))
            print("Restore", variable.op.name)
    with tf.control_dependencies(all_assign):
        no_op = tf.no_op()

    def init_fn(scaffold, session):
        _ = scaffold
        session.run(no_op)
    return init_fn


def model_fn(features, labels, mode, params, config):
    features = copy.copy(features)
    # Add graph nodes for images and labels
    images = tf.identity(features.pop("images"), name="Images")
    if labels is not None:
        labels = tf.identity(labels, name="Labels")
    elif "Labels" in features:
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

    # TODO-0: Check ModeKeys.PREDICT
    if args.eval_per_epoch or mode == ModeKeys.EVAL:
        predictions = features
        predictions["labels"] = labels
        predictions.update(model.predictions)
        predictions.update(model.metrics_dict)

    #############################################################################
    kwargs = {"loss": loss,
              "train_op": train_op,
              "predictions": predictions}

    #############################################################################
    # Initialize partial graph variables by init_fn
    # init_fn = init_partial_model(model, args)
    if args.model == "DenseUNet":
        init_fn = init_dense_model()
        kwargs["scaffold"] = tf.train.Scaffold(init_fn=init_fn)

    if args.load_weights is not None:
        init_fn = init_model(model, args)
        kwargs["scaffold"] = tf.train.Scaffold(init_fn=init_fn)

    return tfes.estimator.EstimatorSpec(mode=mode, **kwargs)
