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
import tensorflow as tf     # Tensorflow >= 1.12.0
from pathlib import Path
from tensorflow.python.platform import tf_logging as logging

import config
import models
import solver
import loss_metrics
import input_pipeline
import custom_evaluator
from utils.logger import create_logger
from custom_estimator import CustomEstimator
from custom_evaluator import EvaluateVolume

ModeKeys = tf.estimator.ModeKeys
TF_RANDOM_SEED = 13579


def _get_arguments(argv):
    parser = argparse.ArgumentParser()
    config.add_arguments(parser)
    models.add_arguments(parser)
    solver.add_arguments(parser)
    loss_metrics.add_arguments(parser)
    input_pipeline.add_arguments(parser)
    custom_evaluator.add_arguments(parser)

    args = parser.parse_args(argv[1:])
    config.check_args(args, parser)
    config.fill_default_args(args)

    return args


def _get_session_config():
    sess_cfg = tf.ConfigProto(allow_soft_placement=True)
    sess_cfg.gpu_options.allow_growth = True
    return sess_cfg


def _custom_tf_logger(args):
    # Set custom logger
    log_dir = Path(__file__).parent / "logs"
    if not log_dir.exists():
        log_dir.mkdir(exist_ok=True)

    log_file = log_dir / "{}_{}".format(args.mode, args.tag)
    logging._logger = create_logger(log_file=log_file, with_time=True,
                                    clear_exist_handlers=True, name="tensorflow")


def main(argv):
    args = _get_arguments(argv)
    _custom_tf_logger(args)
    logging.info(args)

    if args.mode == ModeKeys.TRAIN:
        run_config = tf.estimator.RunConfig(
            tf_random_seed=TF_RANDOM_SEED,
            save_summary_steps=200,
            save_checkpoints_steps=5000,
            session_config=_get_session_config(),
            keep_checkpoint_max=1,
            log_step_count_steps=500,
        )

        params = {"args": args}
        params.update(models.get_model_params(args))
        params.update(solver.get_solver_params(args))
        if not args.train_without_eval:
            params.update(custom_evaluator.get_eval_params(eval_steps=args.eval_steps,
                                                           primary_metric=args.primary_metric,
                                                           secondary_metric=args.secondary_metric))

        estimator = CustomEstimator(models.model_fn, args.model_dir, run_config, params)

        steps, max_steps = ((args.num_of_steps, None)
                            if args.num_of_steps > 0 else (None, args.num_of_total_steps))
        estimator.train(input_pipeline.input_fn,
                        steps=steps,
                        max_steps=max_steps,
                        save_best_ckpt=args.save_best)

    elif args.mode == ModeKeys.EVAL:
        run_config = tf.estimator.RunConfig(
            tf_random_seed=TF_RANDOM_SEED,
            session_config=_get_session_config()
        )

        params = {"args": args}
        params.update(models.get_model_params(args))

        estimator = CustomEstimator(models.model_fn, args.model_dir, run_config, params)

        predict_keys = None
        evaluator = EvaluateVolume(estimator, predict_keys)

        estimator.evaluate(evaluator,
                           input_pipeline.input_fn,
                           checkpoint_path=args.ckpt_path,
                           latest_filename=("checkpoint_best" if not args.eval_final else None))


if __name__ == "__main__":
    tf.app.run(main)
