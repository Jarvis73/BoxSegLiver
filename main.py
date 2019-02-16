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

import argparse
import tensorflow as tf

import config
import models
import solver
import input_pipeline
import session_run_helper as training

ModeKeys = tf.estimator.ModeKeys
TF_RANDOM_SEED = 13579


def _get_arguments():
    parser = argparse.ArgumentParser()
    config.add_arguments(parser)
    models.add_arguments(parser)
    input_pipeline.add_arguments(parser)

    args = parser.parse_args()
    config.check_args(args, parser)
    return args


def _get_session_config():
    sess_cfg = tf.ConfigProto(allow_soft_placement=True)
    sess_cfg.gpu_options.allow_growth = True
    return sess_cfg


def main():
    """
    Estimator-defined running hooks:
        - _DatasetInitializerHook
        - NanTensorHook
        - LoggingTensorHook
        - CheckpointSaverHook
        - StepCounterHook
        - SummarySaverHook

    """
    args = _get_arguments()

    if args.mode == ModeKeys.TRAIN:
        run_config = tf.estimator.RunConfig(
            tf_random_seed=TF_RANDOM_SEED,
            save_summary_steps=100,
            save_checkpoints_steps=5000,
            session_config=_get_session_config(),
            keep_checkpoint_max=3,
            log_step_count_steps=200,
        )

        params = {"args": args}
        params.update(models.get_model_params(args))
        params.update(solver.get_solver_params(args))

        # Construct estimator
        model = tf.estimator.Estimator(models.model_fn, args.model_dir, run_config, params)

        # Train the model
        model.train(input_pipeline.input_fn, max_steps=args.num_of_total_steps)

        # Evaluate the model
        # model.evaluate()

        # Train and evaluate the model
        train_spec = tf.estimator.TrainSpec(input_pipeline.input_fn, max_steps=args.num_of_total_steps)
        best_expoter = tf.estimator.BestExporter()
        eval_spec = tf.estimator.EvalSpec(input_pipeline.input_fn, steps=None, name=None,
                                          hooks=None)
        training.train_and_evaluate(model, train_spec, eval_spec)


if __name__ == "__main__":
    main()
