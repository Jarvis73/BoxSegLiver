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

import sys
import argparse
import functools
import tensorflow as tf     # Tensorflow >= 1.13.0
import tensorflow_estimator as tfes
from pathlib import Path
from tensorflow.python.platform import tf_logging as logging

import config
import loss_metrics
from utils.logger import create_logger
from utils import distribution_utils
from core import solver
from core import models
from core import estimator as estimator_lib
from core import hooks
import input_pipeline
from evaluators import evaluator_liver

ModeKeys = tfes.estimator.ModeKeys
TF_RANDOM_SEED = None


def _get_arguments():
    parser = argparse.ArgumentParser()

    config.add_arguments(parser)
    models.add_arguments(parser)
    solver.add_arguments(parser)
    loss_metrics.add_arguments(parser)

    argv = sys.argv
    if argv[1] == "liver":
        input_pipeline.add_arguments(parser)
        evaluator_liver.add_arguments(parser)
    else:
        raise ValueError("First argument must be choose from [liver], got {}".format(argv[1]))

    args = parser.parse_args(argv[2:])
    config.check_args(args, parser)
    config.fill_default_args(args)

    return args, argv[1]


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

    if args.out_file:
        log_file = log_dir / args.out_file
        with_time = False
    else:
        log_file = log_dir / "{}_{}".format(args.mode, args.tag)
        with_time = True
    logging._logger = create_logger(log_file=log_file, with_time=with_time, file_level=1,
                                    clear_exist_handlers=True, name="tensorflow")


def main():
    args, subcommand = _get_arguments()
    _custom_tf_logger(args)
    logging.debug(args)

    session_config = _get_session_config(args)

    distribution_strategy = distribution_utils.get_distribution_strategy(
        distribution_strategy=args.distribution_strategy,
        num_gpus=args.num_gpus,
        num_workers=1,
        all_reduce_alg=args.all_reduce_alg,
        session_config=session_config)

    save_summary_steps = 200
    if args.mode == ModeKeys.TRAIN:
        log_step_count_steps = 500
        run_config = tfes.estimator.RunConfig(
            train_distribute=distribution_strategy,
            tf_random_seed=TF_RANDOM_SEED,
            save_summary_steps=save_summary_steps,
            save_checkpoints_steps=5000,
            session_config=session_config,
            keep_checkpoint_max=1,
            log_step_count_steps=log_step_count_steps,
        )

        params = {"args": args, "save_best_ckpt": args.save_best}
        params.update(models.get_model_params(args,
                                              build_metrics=True,
                                              build_summaries=bool(save_summary_steps)))
        params.update(solver.get_solver_params(args,
                                               warm_up=args.lr_warm_up,
                                               slow_start_step=args.slow_start_step,
                                               slow_start_learning_rate=args.slow_start_lr))

        if args.eval_epoch:
            params["double_dataloader_modes"] = [ModeKeys.TRAIN, "eval_while_train"]

        estimator = estimator_lib.CustomEstimator(models.model_fn, args.model_dir, run_config, params,
                                                  args.warm_start_from)
        train_hooks = []
        if args.eval_epoch:
            evaluator_lib = eval("evaluator_{}".format(subcommand))
            evaluator = evaluator_lib.get_evaluator(args.evaluator, estimator=estimator)
            eval_hook = hooks.EvaluatorHook(evaluator,
                                            checkpoint_dir=estimator.model_dir,
                                            compare_fn=functools.partial(evaluator.compare,
                                                                         args.primary_metric,
                                                                         args.secondary_metric),
                                            tag=args.tag,
                                            save_steps=args.eval_steps)
            train_hooks.append(eval_hook)

        steps, max_steps = ((args.num_of_steps, None)
                            if args.num_of_steps > 0 else (None, args.num_of_total_steps))
        estimator.train(input_pipeline.input_fn,
                        hooks=train_hooks,
                        steps=steps,
                        max_steps=max_steps)

    elif args.mode == ModeKeys.EVAL:
        params = {"args": args}
        params.update(models.get_model_params(args))
        evaluator_lib = eval("evaluator_{}".format(subcommand))
        evaluator = evaluator_lib.get_evaluator(args.evaluator, model_dir=args.model_dir, params=params)
        evaluator.run(input_pipeline.input_fn,
                      checkpoint_path=args.ckpt_path,
                      latest_filename=(args.load_status_file if not args.eval_final else None),
                      cases=args.eval_num,
                      save=args.save_predict)


if __name__ == "__main__":
    main()
