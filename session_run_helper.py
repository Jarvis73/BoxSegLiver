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
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.estimator import training

from evaluate_volume import EvaluateVolume
from utils import summary_kits

_EvalResult = training._EvalResult
_EvalStatus = training._EvalStatus
_assert_eval_spec = training._assert_eval_spec


class Evaluator(training._TrainingExecutor._Evaluator):
    """ Custom evaluate routine for 3D volumes """

    def __init__(self, estimator, eval_spec, max_training_steps):
        self._estimator = estimator
        self._evaluator = EvaluateVolume(estimator, estimator.params["args"].metrics_eval)

        _assert_eval_spec(eval_spec)
        self._eval_spec = eval_spec

        self._is_final_export_triggered = False
        self._previous_ckpt_path = None
        self._last_warning_time = 0
        self._max_training_steps = max_training_steps

        self.writer_eval = tf.summary.FileWriter(estimator.model_dir, filename_suffix="best_results")

    def evaluate_and_export(self):
        """
        Evaluate and (maybe) export the current model.

        Returns:
            A tuple of `EvalResult` instance and the export results.

        Raises:
            RuntimeError: for any unexpected internal error.
            TypeError: if evaluation result has wrong type.
        """
        latest_ckpt_path = self._estimator.latest_checkpoint()
        if not latest_ckpt_path:
            self._log_err_msg('Estimator is not trained yet. Will start an '
                              'evaluation when a checkpoint is ready.')
            return _EvalResult(status=training._EvalStatus.MISSING_CHECKPOINT), []

        if latest_ckpt_path == self._previous_ckpt_path:
            self._log_err_msg(
                'No new checkpoint ready for evaluation. Skip the current '
                'evaluation pass as evaluation results are expected to be same '
                'for the same checkpoint.')
            return _EvalResult(status=_EvalStatus.NO_NEW_CHECKPOINT), []

        metrics = self._evaluator.evaluate(
            input_fn=self._eval_spec.input_fn,
            hooks=self._eval_spec.hooks,
            checkpoint_path=latest_ckpt_path
        )

        # Summary evaluation results
        tags = [tag for tag in metrics.keys() if tag != tf.GraphKeys.GLOBAL_STEP]
        values = [metrics[tag] for tag in tags]
        tags = ["{}/{}".format(self._estimator.params["args"].tag, tag)
                for tag in tags]
        summary_kits.summary_scalar(self.writer_eval, metrics[tf.GraphKeys.GLOBAL_STEP], tags, values)

        # _EvalResult validates the metrics.
        eval_result = _EvalResult(
            status=_EvalStatus.EVALUATED,
            metrics=metrics,
            checkpoint_path=latest_ckpt_path)

        is_the_final_export = (
            eval_result.metrics[ops.GraphKeys.GLOBAL_STEP] >=
            self._max_training_steps if self._max_training_steps else False)
        export_results = self._export_eval_result(eval_result,
                                                  is_the_final_export)

        if is_the_final_export:
            logging.debug('Calling exporter with the `is_the_final_export=True`.')
            self._is_final_export_triggered = True

        self._last_warning_time = 0
        self._previous_ckpt_path = latest_ckpt_path
        return eval_result, export_results


class TrainingExecutor(training._TrainingExecutor):
    def run_local(self):
        """Runs training and evaluation locally (non-distributed)."""
        _assert_eval_spec(self._eval_spec)

        train_hooks = list(self._train_spec.hooks) + list(self._train_hooks)
        logging.info('Start train and evaluate loop. The evaluate will happen '
                     'after every checkpoint. Checkpoint frequency is determined '
                     'based on RunConfig arguments: save_checkpoints_steps {} or '
                     'save_checkpoints_secs {}.'.format(
                        self._estimator.config.save_checkpoints_steps,
                        self._estimator.config.save_checkpoints_secs))

        evaluator = Evaluator(self._estimator, self._eval_spec,
                              self._train_spec.max_steps)

        listener_for_eval = training._NewCheckpointListenerForEvaluate(
            evaluator, self._eval_spec.throttle_secs,
            self._continuous_eval_listener)
        saving_listeners = [listener_for_eval]

        self._estimator.train(
            input_fn=self._train_spec.input_fn,
            max_steps=self._train_spec.max_steps,
            hooks=train_hooks,
            saving_listeners=saving_listeners)

        eval_result = listener_for_eval.eval_result or _EvalResult(
            status=_EvalStatus.MISSING_CHECKPOINT)
        return eval_result.metrics, listener_for_eval.export_results


class BestSaver(tf.estimator.Exporter):
    def name(self):
        return "Best_Saver"

    def export(self, estimator, export_path, checkpoint_path, eval_result,
               is_the_final_export):
        # TODO
        pass


def train_and_evaluate(estimator, train_spec, eval_spec):
    _assert_eval_spec(eval_spec)

    executor = TrainingExecutor(
        estimator=estimator, train_spec=train_spec, eval_spec=eval_spec)

    return executor.run()


def _assert_evaluator(evaluator):
    """Raise error if `evaluator` is not of the right type."""
    if not isinstance(evaluator, EvaluateVolume):
        raise TypeError('`evaluator` must have type `evaluate_volume.EvaluateVolume`. '
                        'Got: {}'.format(type(evaluator)))
