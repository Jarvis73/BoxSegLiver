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

import yaml  # conda install -c conda-forge pyyaml
import tensorflow as tf
from pathlib import Path

from Networks.UNet import UNet

ModeKeys = tf.estimator.ModeKeys

# Available models
MODEL_ZOO = [
    UNet
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
                       default="trunc_norm",
                       choices=["trunc_norm", "xavier"],
                       required=False, help="Model variable initialization method (default: %(default)s)")
    group.add_argument("--normalizer",
                       type=str,
                       default="batch_norm",
                       choices=["batch_norm"],
                       required=False, help="Normalization method (default: %(default)s)")


def get_model_params(args):
    params = {}

    if False:   # Add sophisticated models
        pass
    else:   # Simpler model (only need "args" to initialize)
        params["model"] = eval(args.model)(args)
        if not args.model_config:
            args.model_config = args.model + ".yml"
        model_config_path = Path(__file__).parent / "Networks" / args.model_config
        with model_config_path.open() as f:
            params["model_kwargs"] = yaml.load(f)

    return params


def model_fn(features, labels, mode, params):
    images = tf.identity(features["images"], name="Images")
    if labels is not None:
        labels = tf.identity(labels, name="Labels")
    elif "labels" in features:
        labels = tf.identity(features["labels"], name="Labels")

    loss = None
    train_op = None
    predictions = None

    if mode == ModeKeys.TRAIN:
        inputs = {"images": images, "labels": labels}

        # create model
        model = params["model"]
        model_args = params.get("model_args", ())
        model_kwargs = params.get("model_kwargs", {})
        loss = model(inputs, ModeKeys.TRAIN, *model_args, **model_kwargs)

        # create solver
        solver = params["solver"]
        solver_args = params.get("solver_args", ())
        solver_kwargs = params.get("solver_kwargs", {})
        train_op = solver(loss, *solver_args, **solver_kwargs)

    if mode == ModeKeys.PREDICT:
        inputs = {"images": images}

        # create model
        model = params["model"]
        model_args = params.get("model_args", ())
        model_kwargs = params.get("model_kwargs", {})
        model(inputs, ModeKeys.PREDICT, *model_args, **model_kwargs)

    if mode == ModeKeys.PREDICT or (mode == ModeKeys.TRAIN and not params["args"].train_without_eval):
        obj_tensors = [model.layers[obj + "Pred"] for obj in model.classes if obj != "Background"]
        predictions = {obj.op.name: obj for obj in obj_tensors}

        with tf.name_scope("LabelProcess/"):
            graph = tf.get_default_graph()
            try:
                one_hot_label = graph.get_tensor_by_name("LabelProcess/one_hot:0")
            except KeyError:
                one_hot_label = tf.one_hot(labels, model.num_classes)

            split_labels = []
            try:
                for i in range(model.num_classes):
                    split_labels.append(graph.get_tensor_by_name("LabelProcess/split:{}".format(i)))
            except KeyError:
                split_labels = tf.split(one_hot_label, model.num_classes, axis=-1)

        others = {
            "Labels": split_labels[1:],
            "Names": features["names"],
            "Pads": features["pads"],
            "GlobalStep": tf.train.get_global_step(tf.get_default_graph())
        }
        predictions.update(others)

    #############################################################################
    kwargs = {"loss": loss,
              "train_op": train_op,
              "predictions": predictions}

    if mode == ModeKeys.TRAIN:
        return tf.estimator.EstimatorSpec(mode=ModeKeys.TRAIN, **kwargs)

    if mode == ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=ModeKeys.PREDICT, **kwargs)
