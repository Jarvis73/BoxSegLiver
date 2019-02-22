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

from tensorflow.python.training import session_run_hook
from tensorflow.python.training import basic_session_run_hooks
from tensorflow.python.platform import tf_logging as logging

from custom_evaluator import EvaluateVolume


class IteratorStringHandleHook(session_run_hook.SessionRunHook):
    """ Hook to initialize string handle of Iterator """

    def __init__(self, train_iterator, eval_iterator):
        self._train_iterator = train_iterator
        self._eval_iterator = eval_iterator
        self._train_handle = None
        self._eval_handle = None

    @property
    def train_handle(self):
        return self._train_handle

    @property
    def eval_handle(self):
        return self._eval_handle

    def begin(self):
        self._train_string_handle = self._train_iterator.string_handle()
        self._eval_string_handle = self._eval_iterator.string_handle()

    def after_create_session(self, session, coord):
        del coord
        self._train_handle, self._eval_handle = session.run([self._train_string_handle,
                                                             self._eval_string_handle])


class BestCheckpointSaverHook(session_run_hook.SessionRunHook):
    """ Hook to save best checkpoints """

    def __init__(self,
                 evaluator,
                 checkpoint_dir,
                 compare_fn,
                 save_secs=None,
                 save_steps=None,
                 saver=None,
                 checkpoint_basename="best_model.ckpt",
                 scaffold=None):
        """Initializes a `CheckpointSaverHook`.

        Args:
          evaluator: `` for evaluate model
          checkpoint_dir: `str`, base directory for the checkpoint files.
          compare_fn: `function`, compare function for the better results
          save_secs: `int`, save every N secs.
          save_steps: `int`, save every N steps.
          saver: `Saver` object, used for saving.
          checkpoint_basename: `str`, base name for the checkpoint files.
          scaffold: `Scaffold`, use to get saver object.

        Raises:
          ValueError: One of `save_steps` or `save_secs` should be set.
          ValueError: At most one of `saver` or `scaffold` should be set.
        """
        logging.info("Create BestCheckpointSaverHook.")
        if saver is not None and scaffold is not None:
            raise ValueError("You cannot provide both saver and scaffold.")
        if not isinstance(evaluator, EvaluateVolume):
            raise TypeError("`evaluator` must be an EvaluateVolume instance")
        self._evaluator = evaluator
        self._compare_fn = compare_fn
        self._saver = saver
        self._checkpoint_dir = checkpoint_dir
        self._save_path = os.path.join(checkpoint_dir, checkpoint_basename)
        self._scaffold = scaffold
        self._timer = basic_session_run_hooks.SecondOrStepTimer(every_secs=save_secs,
                                                                every_steps=save_steps)
        self._steps_per_run = 1
        self._better_result = None
        self._need_save = False

    def _set_steps_per_run(self, steps_per_run):
        self._steps_per_run = steps_per_run

    def begin(self):
        self._global_step_tensor = training_util._get_or_create_global_step_read()  # pylint: disable=protected-access
        if self._global_step_tensor is None:
            raise RuntimeError(
                "Global step should be created to use CheckpointSaverHook.")

    def before_run(self, run_context):
        return session_run_hook.SessionRunArgs(self._global_step_tensor)

    def after_run(self, run_context, run_values):
        stale_global_step = run_values.results
        if self._timer.should_trigger_for_step(
                stale_global_step + self._steps_per_run):
            # get the real value after train op.
            global_step = run_context.session.run(self._global_step_tensor)
            if self._timer.should_trigger_for_step(global_step):
                self._timer.update_last_triggered_step(global_step)
                self._evaluate(run_context.session)
                if self._save(run_context.session, global_step):
                    run_context.request_stop()

    def _evaluate(self, session):
        results = self._evaluator.evaluate_with_session(session)

        if not self._better_result or self._compare_fn(results, self._better_result):
            self._better_result = results
            self._need_save = True

    def _save(self, session, step):
        """Saves the better checkpoint, returns should_stop."""
        if not self._need_save:
            return False
        self._need_save = False

        

