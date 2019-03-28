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

import six
import json
import copy
from functools import partial
from pathlib import Path

from tensorflow.python.data import Dataset
from tensorflow.python.data import Iterator
from tensorflow.python.eager import context
from tensorflow.python.estimator import estimator as estimator_lib
from tensorflow.python.estimator import util as estimator_util
from tensorflow.python.estimator import model_fn as model_fn_lib
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import training
from tensorflow.python.training import training_util
from tensorflow.python.training import warm_starting_util
# from tensorflow.python.training import monitored_session
from tensorflow.python.training import checkpoint_management
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import random_seed
from tensorflow.python.framework import errors_impl
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.util import function_utils
# from tensorflow.python.ops import math_ops

from custom_evaluator_base import EvaluateBase
from custom_hooks import IteratorStringHandleHook
from custom_hooks import BestCheckpointSaverHook
from custom_hooks import FeedGuideHook

EVAL_WHILE_TRAIN = "eval_while_train"


def _load_global_step_from_checkpoint_dir(checkpoint_dir):
    try:
        checkpoint_reader = training.NewCheckpointReader(
            training.latest_checkpoint(checkpoint_dir))
        return checkpoint_reader.get_tensor(ops.GraphKeys.GLOBAL_STEP)
    except:
        return 0


def _add_key_value(feed_dict, key, value):
    if feed_dict is None:
        return {key: value}
    elif isinstance(feed_dict, dict):
        feed_dict[key] = value
        return feed_dict
    else:
        raise TypeError("`feed_dict` must be None or a dict instance")


def _check_dataset_structure(ds1, ds2):
    if not isinstance(ds1, Dataset):
        raise TypeError("input_fn must return Dataset instance, but got {}"
                        .format(type(ds1)))
    if not isinstance(ds2, Dataset):
        raise TypeError("input_fn must return Dataset instance, but got {}"
                        .format(type(ds2)))

    # if ds1.output_shapes != ds2.output_shapes:
    #     raise ValueError("Train/Eval dataset shapes mismatch: {} vs {}"
    #                      .format(ds1.output_shapes, ds2.output_shapes))

    if ds1.output_types != ds2.output_types:
        raise ValueError("Train/Eval dataset types mismatch: {} vs {}"
                         .format(ds1.output_types, ds2.output_types))


def parse_input_fn_result(train_with_eval, result, handler=None):
    """Gets features, labels, and hooks from the result of an Estimator input_fn.

    Parameters
    ----------
    train_with_eval: bool
        train with evaluation or not
    result: output of an input_fn to an estimator, which should be one of:

        For train without eval:
            * A 'tf.data.Dataset' object: Outputs of `Dataset` object must be a
            tuple (features, labels) with same constraints as below.
            * A tuple (features, labels): Where `features` is a `Tensor` or a
            dictionary of string feature name to `Tensor` and `labels` is a
            `Tensor` or a dictionary of string label name to `Tensor`. Both
            `features` and `labels` are consumed by `model_fn`. They should
            satisfy the expectation of `model_fn` from inputs.
        For train with eval:
            * A list of 2 `tf.data.Dataset` object: Train Dataset & Eval Dataset
    handler: placeholder

    Returns
    -------
    Tuple of features, labels, and input_hooks, where features are as described
    above, labels are as described above or None, and input_hooks are a list
    of SessionRunHooks to be included when running.

    Raises:
        ValueError: if the result is a list or tuple of length != 2.
    """
    input_hooks = []
    with ops.name_scope("DataGenerator"):
        if not train_with_eval:
            try:
                # We can't just check whether this is a tf.data.Dataset instance here,
                # as this is plausibly a PerDeviceDataset. Try treating as a dataset first.
                iterator = result.make_initializable_iterator()
            except AttributeError:
                # Not a dataset or dataset-like-object. Move along.
                pass
            else:
                input_hooks.append(estimator_util._DatasetInitializerHook(iterator))
                result = iterator.get_next()
                return estimator_util.parse_iterator_result(result) + (input_hooks,)
        else:
            err_str = "`result` must be a list of Dataset instance if set train_with_eval as True"
            if not isinstance(result, (list, tuple)):
                raise TypeError(err_str)
            if len(result) != 2:
                raise ValueError("`result` should contains 2 Dataset instances, but got {}".format(len(result)))
            _check_dataset_structure(result[0], result[1])

            train_iterator = result[0].make_one_shot_iterator()
            eval_iterator = result[1].make_initializable_iterator()
            input_hooks.append(estimator_util._DatasetInitializerHook(eval_iterator))

            iterator = Iterator.from_string_handle(handler, result[0].output_types, result[0].output_shapes,
                                                   result[0].output_classes)
            result = iterator.get_next()
            return estimator_util.parse_iterator_result(result) + (train_iterator, eval_iterator, input_hooks)


class CustomEstimator(object):
    """ Custom Estimator

    Predefined hooks:
        StopAtStepHook
        NanTensorHook
        _DatasetInitializerHook
        LoggingTensorHook
        StepCounterHook
        SummarySaverHook
        IteratorStringHandleHook(custom_hooks): initialize iterator string handle for switching multiple iterator
        BestCheckpointSaverHook(custom_hooks): save best checkpoints
        CheckpointInputPipelineHook(tf.data.experiment): save input pipeline
    """
    def __init__(self, model_fn, model_dir=None, config=None, params=None, warm_start_from=None):
        self._config = estimator_lib.maybe_overwrite_model_dir_and_session_config(config, model_dir)
        # Model directory
        self._model_dir = self._config.model_dir
        self._session_config = self._config.session_config
        logging.info('Using config: %s', str(vars(self._config)))

        if model_fn is None:
            raise ValueError('model_fn must be provided to Estimator.')
        self._model_fn = model_fn
        # We shouldn't deep copy params because of class instances(like model, solver instances) in its values
        self._params = params or {}

        self._feed_dict = None
        self._warm_start_settings = estimator_lib._get_default_warm_start_settings(
            warm_start_from)

        self._train_with_eval = not self._params["args"].train_without_eval
        self._predict_keys = None
        self.dataset_handle_hook = None

    @property
    def model_dir(self):
        return self._model_dir

    @property
    def config(self):
        return copy.deepcopy(self._config)

    @property
    def params(self):
        return self._params

    @property
    def model_fn(self):
        def public_model_fn(features, labels, mode, config):
            return self._call_model_fn(features, labels, mode, config)

        return public_model_fn

    def train(self,
              input_fn,
              steps=None,
              hooks=None,
              max_steps=None,
              saving_listeners=None,
              save_best_ckpt=False):
        with context.graph_mode():
            if (steps is not None) and (max_steps is not None):
                raise ValueError('Can not provide both steps and max_steps.')
            if steps is not None and steps <= 0:
                raise ValueError('Must specify steps > 0, given: {}'.format(steps))
            if max_steps is not None and max_steps <= 0:
                raise ValueError(
                    'Must specify max_steps > 0, given: {}'.format(max_steps))

            if max_steps is not None:
                start_step = _load_global_step_from_checkpoint_dir(self._model_dir)
                if max_steps <= start_step:
                    logging.info('Skipping training since max_steps has already saved.')
                    return self

            hooks = estimator_lib._check_hooks_type(hooks)
            hooks.append(training.StopAtStepHook(steps, max_steps))

            saving_listeners = estimator_lib._check_listeners_type(saving_listeners)
            loss = self._train_model(input_fn, hooks, saving_listeners, save_best_ckpt)
            logging.info('Loss for final step: %s.', loss)
            return self

    def evaluate(self,
                 evaluator,
                 input_fn,
                 hooks=None,
                 checkpoint_path=None,
                 latest_filename=None,
                 cases=None):
        if not isinstance(evaluator, EvaluateBase):
            raise TypeError("`evaluator` must be a EvaluateBase instance")
        checkpoint_path = self._checkpoint_path(checkpoint_path, latest_filename)
        results = evaluator.evaluate(input_fn,
                                     hooks=hooks,
                                     checkpoint_path=checkpoint_path,
                                     cases=cases)
        with (Path(self.model_dir) / "eval_results.txt").open("w") as f:
            json.dump(results, f)

    def predict(self,
                input_fn,
                predict_keys=None,
                hooks=None,
                checkpoint_path=None,
                latest_filename=None,
                yield_single_examples=True):
        hooks = estimator_lib._check_hooks_type(hooks)

        checkpoint_path = self._checkpoint_path(checkpoint_path, latest_filename)

        with ops.Graph().as_default() as g:
            random_seed.set_random_seed(self._config.tf_random_seed)
            self._create_and_assert_global_step(g)
            features, labels, input_hooks = self._get_features_and_labels_from_input_fn(
                input_fn, model_fn_lib.ModeKeys.EVAL)

            estimator_spec = self._call_model_fn(
                features, labels, model_fn_lib.ModeKeys.PREDICT, self.config)

            # Evaluating volume don't need metrics in model, we use XXXPred to generate 3D predict
            if not predict_keys:
                predict_keys = [x for x in estimator_spec.predictions
                                if x not in self.params["model"].metrics_dict]
                predict_keys.extend(list(self.params["model"].metrics_eval))
            predictions = self._extract_keys(estimator_spec.predictions, predict_keys)
            all_hooks = list(input_hooks)
            all_hooks.extend(hooks)
            all_hooks.extend(list(estimator_spec.prediction_hooks or []))

            with training.MonitoredSession(
                    session_creator=training.ChiefSessionCreator(
                        checkpoint_filename_with_path=checkpoint_path,
                        master=self._config.master,
                        scaffold=estimator_spec.scaffold,
                        config=self._session_config),
                    hooks=all_hooks) as mon_sess:
                while not mon_sess.should_stop():
                    preds_evaluated = mon_sess.run(predictions)
                    if not yield_single_examples:
                        yield preds_evaluated
                    elif not isinstance(predictions, dict):
                        for pred in preds_evaluated:
                            yield pred
                    else:
                        for i in range(self._extract_batch_length(preds_evaluated)):
                            yield {key: value[i] for key, value in six.iteritems(preds_evaluated)}

    def predict_with_guide(self,
                           input_fn,
                           predict_keys=None,
                           hooks=None,
                           checkpoint_path=None,
                           latest_filename=None,
                           yield_single_examples=True):
        hooks = estimator_lib._check_hooks_type(hooks)

        checkpoint_path = self._checkpoint_path(checkpoint_path, latest_filename)

        with ops.Graph().as_default() as g:
            random_seed.set_random_seed(self._config.tf_random_seed)
            self._create_and_assert_global_step(g)
            features, labels, input_hooks = self._get_features_and_labels_from_input_fn(
                input_fn, model_fn_lib.ModeKeys.EVAL)

            features_ph = {key: array_ops.placeholder(value.dtype, value.shape, name=key)
                           for key, value in features.items()}
            labels_ph = array_ops.placeholder(labels.dtype, labels.shape, name="labels")
            feed_guide_hook = FeedGuideHook(features_ph, labels_ph, features, labels,
                                            self.model_dir)

            estimator_spec = self._call_model_fn(
                features_ph, labels_ph, model_fn_lib.ModeKeys.PREDICT, self.config)

            # Evaluating volume don't need metrics in model, we use XXXPred to generate 3D predict
            if not predict_keys:
                predict_keys = [x for x in estimator_spec.predictions
                                if x not in self.params["model"].metrics_dict]
            predictions = self._extract_keys(estimator_spec.predictions, predict_keys)
            feed_guide_hook.predictions = predictions

            all_hooks = list(input_hooks) + [feed_guide_hook]
            all_hooks.extend(hooks)
            all_hooks.extend(list(estimator_spec.prediction_hooks or []))

            with training.MonitoredSession(
                    session_creator=training.ChiefSessionCreator(
                        checkpoint_filename_with_path=checkpoint_path,
                        master=self._config.master,
                        scaffold=estimator_spec.scaffold,
                        config=self._session_config),
                    hooks=all_hooks) as mon_sess:
                while not mon_sess.should_stop():
                    preds_evaluated = mon_sess.run(predictions)
                    if not yield_single_examples:
                        yield preds_evaluated
                    elif not isinstance(predictions, dict):
                        for pred in preds_evaluated:
                            yield pred
                    else:
                        for i in range(self._extract_batch_length(preds_evaluated)):
                            yield {key: value[i] for key, value in six.iteritems(preds_evaluated)}

    def predict_with_session(self, session, predict_keys=None, steps=None, yield_single_examples=False):
        predictions = self._extract_keys(self._predict_keys, predict_keys)
        pred_feed_dict = self._params["model"].get_eval_feed_dict()
        if self._train_with_eval:
            pred_feed_dict = _add_key_value(pred_feed_dict, self.handler, self.dataset_handle_hook.eval_handle)

        try:
            # Initialize evaluation iterator
            session.run(self.eval_iterator.initializer)

            counter = 0
            while True:
                if steps and counter >= steps:
                    break
                preds_evaluated = session.run(predictions, pred_feed_dict)
                if not yield_single_examples:
                    yield preds_evaluated
                elif not isinstance(predictions, dict):
                    for pred in preds_evaluated:
                        yield pred
                else:
                    for i in range(self._extract_batch_length(preds_evaluated)):
                        yield {key: value[i] for key, value in preds_evaluated.items()}
                counter += 1
        except errors_impl.OutOfRangeError:
            pass

    def _checkpoint_path(self, checkpoint_path, latest_filename):
        if not checkpoint_path:
            latest_path = checkpoint_management.latest_checkpoint(
                self._model_dir, latest_filename)
            if not latest_path:
                logging.info('Could not find trained model in model_dir: {}, running '
                             'initialization to evaluate.'.format(self._model_dir))
            checkpoint_path = latest_path
        return checkpoint_path

    @staticmethod
    def _extract_keys(predictions, predict_keys):
        """Extracts `predict_keys` from `predictions`."""
        if not predict_keys:
            return predictions
        if not isinstance(predictions, dict):
            raise ValueError(
                'predict_keys argument is not valid in case of non-dict predictions.')
        existing_keys = predictions.keys()
        predictions = {
            key: value
            for key, value in six.iteritems(predictions) if key in predict_keys
        }
        if not predictions:
            raise ValueError('Expected to run at least one output from %s, '
                             'provided %s.' % (existing_keys, predict_keys))
        return predictions

    @staticmethod
    def _extract_batch_length(preds_evaluated):
        """Extracts batch length of predictions."""
        batch_length = None
        for key, value in preds_evaluated.items():
            batch_length = batch_length or value.shape[0]
            if value.shape[0] != batch_length:
                raise ValueError('Batch length of predictions should be same. %s has '
                                 'different batch length than others.' % key)
        return batch_length

    def _create_and_assert_global_step(self, graph):
        """Creates and asserts properties of the global step.

        Args:
            graph: The graph in which to create the global step tensor.

        Returns:
            The global step `tf.Tensor`.
        """

        with variable_scope.variable_scope(self.params["args"].tag):
            step = training.create_global_step(graph)
        assert step.dtype.is_integer
        return step

    def _call_input_fn(self, input_fn, mode):
        """Calls the input function.

        Args:
          input_fn: The input function.
          mode: `tf.estimator.ModeKeys`

        Returns:
          The return value of the passed `input_fn`, which should be one of:

            * A 'tf.data.Dataset' object: Outputs of `Dataset` object must be a
                tuple `(features, labels)` with same constraints as below.
            * A tuple `(features, labels)`: Where `features` is a `Tensor` or a
              dictionary of string feature name to `Tensor` and `labels` is a
              `Tensor` or a dictionary of string label name to `Tensor`. Both
              `features` and `labels` are consumed by `model_fn`. They should
              satisfy the expectation of `model_fn` from inputs.

        Raises:
          ValueError: if `input_fn` takes invalid arguments.
        """
        input_fn_args = function_utils.fn_args(input_fn)
        kwargs = {}
        if 'mode' in input_fn_args:
            kwargs['mode'] = mode
        if 'params' in input_fn_args:
            kwargs['params'] = self.params
        if 'config' in input_fn_args:
            kwargs['config'] = self.config
        with ops.device('/cpu:0'):
            return input_fn(**kwargs)

    def _get_features_and_labels_from_input_fn(self, input_fn, mode):
        """Extracts the `features` and labels from return values of `input_fn`."""
        return parse_input_fn_result(False, self._call_input_fn(input_fn, mode))

    def _get_features_and_labels_for_train_and_eval(self, input_fn, handler):
        return parse_input_fn_result(
            True,
            [self._call_input_fn(input_fn, model_fn_lib.ModeKeys.TRAIN),  # Keep order: [train, eval]
             self._call_input_fn(input_fn, EVAL_WHILE_TRAIN)],
            handler)

    def _train_model(self, input_fn, hooks, saving_listeners, save_best_ckpt):
        """Initiate training with `input_fn`, without `DistributionStrategies`.

        Args:
            input_fn: A function that provides input data for training as mini-batches.
            hooks: List of `tf.train.SessionRunHook` subclass instances. Used for
            callbacks inside the training loop.
            saving_listeners: list of `tf.train.CheckpointSaverListener` objects. Used
            for callbacks that run immediately before or after checkpoint savings.
            save_best_ckpt: boolean

        Returns:
            Loss from training
        """
        worker_hooks = []
        worker_hooks.extend(hooks)
        with ops.Graph().as_default() as g:     # Set device if distribution
            random_seed.set_random_seed(self._config.tf_random_seed)
            global_step_tensor = self._create_and_assert_global_step(g)
            training_util._get_or_create_global_step_read(g)

            if self._train_with_eval:
                self.handler = array_ops.placeholder(dtypes.string, shape=(), name="Handler")
                features, labels, self.train_iterator, self.eval_iterator, input_hooks = (
                    self._get_features_and_labels_for_train_and_eval(
                        input_fn, self.handler))
            else:
                self.handler, self.train_iterator, self.eval_iterator = None, None, None
                features, labels, input_hooks = (
                    self._get_features_and_labels_from_input_fn(
                        input_fn, model_fn_lib.ModeKeys.TRAIN))

            worker_hooks.extend(input_hooks)

            estimator_spec = self._call_model_fn(features, labels, model_fn_lib.ModeKeys.TRAIN, self.config)
            self._feed_dict = self._params["model"].feed_dict

            return self._train_with_estimator_spec(estimator_spec, worker_hooks,
                                                   global_step_tensor, saving_listeners,
                                                   save_best_ckpt)

    def _call_model_fn(self, features, labels, mode, config):
        model_fn_args = function_utils.fn_args(self._model_fn)
        kwargs = {}
        if 'labels' in model_fn_args:
            kwargs['labels'] = labels
        else:
            if labels is not None:
                raise ValueError(
                    'model_fn does not take labels, but input_fn returns labels.')
        if 'mode' in model_fn_args:
            kwargs['mode'] = mode
        if 'params' in model_fn_args:
            kwargs['params'] = self.params
        if 'config' in model_fn_args:
            kwargs['config'] = config

        logging.info('Calling model_fn.')
        model_fn_results = self._model_fn(features=features, **kwargs)
        logging.info('Done calling model_fn.')

        if not isinstance(model_fn_results, model_fn_lib.EstimatorSpec):
            raise ValueError('model_fn should return an EstimatorSpec.')

        return model_fn_results

    def _train_with_estimator_spec(self, estimator_spec, worker_hooks,
                                   global_step_tensor, saving_listeners, save_best_ckpt):
        """Train a model with the given Estimator Spec."""
        if self._warm_start_settings:
            logging.info('Warm-starting with WarmStartSettings: %s' %
                         (self._warm_start_settings,))
            warm_starting_util.warm_start(*self._warm_start_settings)

        worker_hooks.append(training.NanTensorHook(estimator_spec.loss))
        if self._config.log_step_count_steps is not None:
            tensors = {"loss": estimator_spec.loss,
                       "step": global_step_tensor}
            tensors.update(self._params["model"].metrics)
            worker_hooks.append(
                training.LoggingTensorHook(tensors, every_n_iter=self._config.log_step_count_steps))
        worker_hooks.extend(estimator_spec.training_hooks)

        # Create Saver object
        if not (estimator_spec.scaffold.saver or ops.get_collection(ops.GraphKeys.SAVERS)):
            ops.add_to_collection(
                ops.GraphKeys.SAVERS,
                training.Saver(
                    sharded=True,
                    max_to_keep=self._config.keep_checkpoint_max,
                    keep_checkpoint_every_n_hours=(
                        self._config.keep_checkpoint_every_n_hours),
                    defer_build=True,
                    save_relative_paths=True))

        chief_hooks = []
        all_hooks = worker_hooks + list(estimator_spec.training_chief_hooks)
        saver_hooks = [
            h for h in all_hooks if isinstance(h, training.CheckpointSaverHook)]
        if (self._config.save_checkpoints_secs or
                self._config.save_checkpoints_steps):
            if not saver_hooks:
                chief_hooks = [
                    training.CheckpointSaverHook(
                        self._model_dir,
                        save_secs=self._config.save_checkpoints_secs,
                        save_steps=self._config.save_checkpoints_steps,
                        scaffold=estimator_spec.scaffold)
                ]
                saver_hooks = [chief_hooks[0]]
        if saving_listeners:
            if not saver_hooks:
                raise ValueError(
                    'There should be a CheckpointSaverHook to use saving_listeners. '
                    'Please set one of the RunConfig.save_checkpoints_steps or '
                    'RunConfig.save_checkpoints_secs.')
            else:
                # It is expected to have one CheckpointSaverHook. If multiple, we pick
                # up the first one to add listener.
                saver_hooks[0]._listeners.extend(saving_listeners)  # pylint: disable=protected-access

        if self._train_with_eval:
            self.dataset_handle_hook = IteratorStringHandleHook(self.train_iterator,
                                                                self.eval_iterator)
            worker_hooks.append(self.dataset_handle_hook)
            self._predict_keys = estimator_spec.predictions

        if save_best_ckpt:
            EvaluatorCls = self._params.get("evaluator", None)
            if not issubclass(EvaluatorCls, EvaluateBase):
                raise TypeError("Parameter `evaluator` must be a EvaluateBase instance, but got {}"
                                .format(type(EvaluatorCls)))
            eval_kwargs = self._params.get("eval_kwargs", {})
            eval_steps = self._params.get("eval_steps", 2500)
            primary_metric = self._params.get("primary_metric", None)
            secondary_metric = self._params.get("secondary_metric", None)

            # We must construct Evaluator inside a graph scope
            evaluator = EvaluatorCls(self, **eval_kwargs)

            worker_hooks.append(
                BestCheckpointSaverHook(
                    evaluator=evaluator,
                    checkpoint_dir=self._model_dir,
                    compare_fn=partial(evaluator.compare,
                                       primary_metric=primary_metric,
                                       secondary_metric=secondary_metric),
                    tag=self._params["args"].tag,
                    save_steps=eval_steps))

        # Training session monitor
        with training.MonitoredTrainingSession(
                master=self._config.master,
                is_chief=self._config.is_chief,
                checkpoint_dir=self._model_dir,
                scaffold=estimator_spec.scaffold,
                hooks=worker_hooks,
                chief_only_hooks=(
                    tuple(chief_hooks) + tuple(estimator_spec.training_chief_hooks)),
                save_checkpoint_secs=0,
                save_summaries_steps=self._config.save_summary_steps,
                config=self._session_config,
                log_step_count_steps=self._config.log_step_count_steps) as mon_sess:
            loss = None

            # Make sure that use self.dataset_handle_hook.xxx_handle after create MonitoredSession()
            self._feed_dict = _add_key_value(self._feed_dict,
                                             self.handler, self.dataset_handle_hook.train_handle)
            while not mon_sess.should_stop():
                _, loss = mon_sess.run([estimator_spec.train_op, estimator_spec.loss], self._feed_dict)
            return loss

    def _evaluate_build_graph(self, input_fn):
        """Builds the graph and related hooks to run evaluation."""
        random_seed.set_random_seed(self._config.tf_random_seed)
        self._create_and_assert_global_step(ops.get_default_graph())

        features, labels, input_hooks = self._get_features_and_labels_from_input_fn(
            input_fn, model_fn_lib.ModeKeys.EVAL)

        estimator_spec = self._call_model_fn(
            features, labels, model_fn_lib.ModeKeys.EVAL, self.config)

        return estimator_spec
