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
import tensorflow as tf     # Tensorflow >= 1.13.0
import tensorflow_estimator as tfes
from pathlib import Path
from tensorflow.python.platform import tf_logging as logging

import config
import models
import solver
import loss_metrics
import input_pipeline_nf
import custom_evaluator
from utils.logger import create_logger
from utils import distribution_utils
from custom_estimator import CustomEstimator
from custom_hooks import LogLearningRateHook

ModeKeys = tfes.estimator.ModeKeys
TF_RANDOM_SEED = None
input_pipeline = input_pipeline_nf


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


def _get_session_config(args):
    if args.device_mem_frac > 0:
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.device_mem_frac)
    else:
        gpu_options = tf.GPUOptions(allow_growth=True)
    sess_cfg = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)
    # sess_cfg = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options,
    #                           device_count={"CPU": 1},
    #                           inter_op_parallelism_threads=1,
    #                           intra_op_parallelism_threads=20)
    return sess_cfg


def _custom_tf_logger(args):
    # Set custom logger
    log_dir = Path(args.model_dir) / "logs"
    if not log_dir.exists():
        log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / "{}_{}".format(args.mode, args.tag)
    logging._logger = create_logger(log_file=log_file, with_time=True, file_level=1,
                                    clear_exist_handlers=True, name="tensorflow")


def main(argv):
    args = _get_arguments(argv)
    _custom_tf_logger(args)
    logging.debug(args)

    session_config = _get_session_config(args)

    distribution_strategy = distribution_utils.get_distribution_strategy(
        distribution_strategy=args.distribution_strategy,
        num_gpus=args.num_gpus,
        num_workers=1,
        all_reduce_alg=args.all_reduce_alg,
        session_config=session_config)

    if args.mode == ModeKeys.TRAIN:
        log_step_count_steps = 500
        run_config = tfes.estimator.RunConfig(
            train_distribute=distribution_strategy,
            tf_random_seed=TF_RANDOM_SEED,
            save_summary_steps=200,
            save_checkpoints_steps=5000,
            session_config=session_config,
            keep_checkpoint_max=1,
            log_step_count_steps=log_step_count_steps,
        )

        params = {"args": args}
        params.update(models.get_model_params(args))
        params.update(solver.get_solver_params(args,
                                               warm_up=args.lr_warm_up,
                                               slow_start_step=args.slow_start_step,
                                               slow_start_learning_rate=args.slow_start_lr))
        if not args.train_without_eval:
            params.update(custom_evaluator.get_eval_params(evaluator=args.evaluator,
                                                           eval_steps=args.eval_steps,
                                                           largest=True,
                                                           merge_tumor_to_liver=True,
                                                           primary_metric=args.primary_metric,
                                                           secondary_metric=args.secondary_metric))

        estimator = CustomEstimator(models.model_fn, args.model_dir, run_config, params,
                                    args.warm_start_from)

        steps, max_steps = ((args.num_of_steps, None)
                            if args.num_of_steps > 0 else (None, args.num_of_total_steps))
        estimator.train(input_pipeline.input_fn,
                        hooks=[LogLearningRateHook(tag=args.tag,
                                                   every_n_steps=log_step_count_steps,
                                                   output_dir=args.model_dir)],
                        steps=steps,
                        max_steps=max_steps,
                        save_best_ckpt=args.save_best)

    elif args.mode == ModeKeys.EVAL:
        run_config = tfes.estimator.RunConfig(
            tf_random_seed=TF_RANDOM_SEED,
            session_config=_get_session_config(args)
        )

        params = {"args": args}
        params.update(models.get_model_params(args))
        eval_params = custom_evaluator.get_eval_params(evaluator=args.evaluator,
                                                       eval_steps=args.eval_steps,
                                                       largest=True,
                                                       merge_tumor_to_liver=True)

        estimator = CustomEstimator(models.model_fn, args.model_dir, run_config, params)

        evaluator = eval_params["evaluator"](estimator, **eval_params["eval_kwargs"])
        estimator.evaluate(evaluator,
                           input_pipeline.input_fn,
                           checkpoint_path=args.ckpt_path,
                           latest_filename=(args.load_status_file if not args.eval_final else None),
                           cases=args.eval_num)


if __name__ == "__main__":
    tf.app.run(main)