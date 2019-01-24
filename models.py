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

ModeKeys = tf.estimator.ModeKeys


def model_fn(features, labels, mode, params):
    if mode == ModeKeys.TRAIN:
        inputs = {"images": features, "labels": labels}

        # create model
        model = params["model"]
        model_args = params.get("model_args", ())
        model_kwargs = params.get("model_kwargs", {})
        loss = model(inputs, mode, *model_args, **model_kwargs)

        # create solver
        solver = params["solver"]
        solver_args = params.get("solver_args", ())
        solver_kwargs = params.get("solver_kwargs", {})
        train_op = solver(loss, *solver_args, **solver_kwargs)

        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss,
            train_op=train_op
        )
