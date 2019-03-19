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

import os
import json
import numpy as np
from pathlib import Path

from tensorflow.python.framework import ops
# from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
# from tensorflow.python.framework import meta_graph
# from tensorflow.python.ops import array_ops
# from tensorflow.python.ops import state_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import training_util
from tensorflow.python.training import session_run_hook
from tensorflow.python.training import basic_session_run_hooks
from tensorflow.python.training import saver as saver_lib
from tensorflow.python.training.summary_io import SummaryWriterCache

import data_kits.build_data as data_ops
from custom_evaluator_base import EvaluateBase
from utils.summary_kits import summary_scalar
from utils.array_kits import get_gd_image_multi_objs
from config import CustomKeys


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
        logging.info("Initialize Dataset.string_handle done!")


class BestCheckpointSaverHook(session_run_hook.SessionRunHook):
    """ Hook to save best checkpoints """

    def __init__(self,
                 evaluator,
                 checkpoint_dir,
                 compare_fn,
                 tag=None,
                 save_secs=None,
                 save_steps=None,
                 saver=None,
                 checkpoint_basename="best_model.ckpt"):
        """Initializes a `CheckpointSaverHook`.

        Args:
          tag: `str`, model tag
          evaluator: for evaluate model
          checkpoint_dir: `str`, base directory for the checkpoint files.
          compare_fn: `function`, compare function for the better results
          save_secs: `int`, save every N secs.
          save_steps: `int`, save every N steps.
          saver: `Saver` object, used for saving.
          checkpoint_basename: `str`, base name for the checkpoint files.

        Raises:
          ValueError: One of `save_steps` or `save_secs` should be set.
          ValueError: At most one of `saver` or `scaffold` should be set.
        """
        logging.info("Create BestCheckpointSaverHook.")
        if not isinstance(evaluator, EvaluateBase):
            raise TypeError("`evaluator` must be an EvaluateBase instance")
        self._summary_tag = tag + "/Eval/{}" if tag else "Eval/{}"
        self._evaluator = evaluator
        self._compare_fn = compare_fn
        self._saver = saver
        self._checkpoint_dir = checkpoint_dir
        self._save_path = os.path.join(checkpoint_dir, checkpoint_basename)
        self._timer = basic_session_run_hooks.SecondOrStepTimer(every_secs=save_secs,
                                                                every_steps=save_steps)
        self._steps_per_run = 1
        self._need_save = False

        self._better_result = None
        if self._get_best_result_dump_file().exists():
            with self._get_best_result_dump_file().open() as f:
                self._better_result = json.load(f)
            logging.info("Best result records loaded!")

    def _set_steps_per_run(self, steps_per_run):
        self._steps_per_run = steps_per_run

    def begin(self):
        self._summary_writer = SummaryWriterCache.get(self._checkpoint_dir)
        self._global_step_tensor = training_util._get_or_create_global_step_read()  # pylint: disable=protected-access
        if self._global_step_tensor is None:
            raise RuntimeError(
                "Global step should be created to use BestCheckpointSaverHook.")

    # def after_create_session(self, session, coord):
    #     global_step = session.run(self._global_step_tensor)
    #
    #     # We do write graph and saver_def at the first call of before_run.
    #     # We cannot do this in begin, since we let other hooks to change graph and
    #     # add variables in begin. Graph is finalized after all begin calls.
    #     training_util.write_graph(
    #         ops.get_default_graph().as_graph_def(add_shapes=True),
    #         self._checkpoint_dir,
    #         "best_graph.pbtxt")
    #     # The checkpoint saved here is the state at step "global_step".
    #     self._evaluate(session, global_step)
    #     self._timer.update_last_triggered_step(global_step)

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
                if self._evaluate(run_context.session, global_step):
                    run_context.request_stop()

    def end(self, session):
        last_step = session.run(self._global_step_tensor)
        if last_step != self._timer.last_triggered_step():
            self._evaluate(session, last_step)

    def _evaluate(self, session, step):
        results = self._evaluator.evaluate_with_session(session)

        if not self._better_result or self._compare_fn(results, self._better_result):
            self._better_result = results
            self._need_save = True

        self._summary(step, results)
        return self._save(session, step)

    def _save(self, session, step):
        """Saves the better checkpoint, returns should_stop."""
        if not self._need_save:
            return False
        self._need_save = False
        logging.info("Saving (best) checkpoints for %d into %s.", step, self._save_path)

        # We must use a different latest_filename comparing with the default "checkpoint"
        self._get_saver().save(session, self._save_path, global_step=step,
                               latest_filename="checkpoint_best")
        with self._get_best_result_dump_file().open("w") as f:
            json.dump(self._get_result_for_json_dump(), f)

        should_stop = False
        return should_stop

    def _summary(self, step, result=None):
        if result is None:
            result = self._better_result

        tags, values = [], []
        for key, value in result.items():
            if key == ops.GraphKeys.GLOBAL_STEP:
                continue
            tags.append(self._summary_tag.format(key))
            values.append(value)

        summary_scalar(self._summary_writer, step, tags, values)

    def _get_result_for_json_dump(self):
        res = {}
        for key, val in self._better_result.items():
            if isinstance(val, np.int64):
                val = int(val)
            else:
                val = float(val)
            res[key] = val

        return res

    def _get_best_result_dump_file(self, name="best_result"):
        return Path(self._save_path).parent / name

    def _get_saver(self):
        if self._saver is not None:
            return self._saver

        # Get saver from the SAVERS collection if present.
        collection_key = ops.GraphKeys.SAVERS
        savers = ops.get_collection(collection_key)
        if not savers:
            raise RuntimeError(
                "No items in collection {}. Please add a saver to the collection "
                "or provide a saver or scaffold.".format(collection_key))
        elif len(savers) > 1:
            raise RuntimeError(
                "More than one item in collection {}. "
                "Please indicate which one to use by passing it to the constructor."
                .format(collection_key))

        # We create a new saver with the SaverDef of the model main saver
        # With SaverDef we don't need create extra graph nodes
        # It is pity that parameter `max_to_keep` is saved to SaverDef and we cannot
        # change it in duplicate Saver
        self._saver = saver_lib.Saver(saver_def=savers[0].as_saver_def())
        return self._saver


class FeedGuideHook(session_run_hook.SessionRunHook):
    def __init__(self, features_ph, labels_ph, features, labels, model_dir):
        self.features_ph = features_ph
        self.labels_ph = labels_ph
        self.features = features
        self.labels = labels
        self.features_val = None
        self.labels_val = None
        self.predictions = None
        self.first = True
        self.guides = []
        self.cur_case = None
        self.bbox = None
        self.img_reader = data_ops.ImageReader()
        self.model_dir = model_dir

    def before_run(self, run_context):
        if self.first:
            self.features_val, self.labels_val = run_context.session.run(
                [self.features, self.labels])
            self.first = False
            self.guides.append(self.features_val["images"][0, ..., -1])
            self.cur_case = self.features_val["names"][0].decode("utf-8")
            self.bbox = self.features_val["bboxes"][0]

        feed_dict = {value: self.features_val[key] for key, value in self.features_ph.items()}
        feed_dict[self.labels_ph] = self.labels_val

        return session_run_hook.SessionRunArgs(self.predictions, feed_dict=feed_dict)

    def after_run(self, run_context, run_values):
        predictions = run_values.results

        try:
            self.features_val, self.labels_val = run_context.session.run(
                [self.features, self.labels])
        except errors_impl.OutOfRangeError:
            self._save_guide()
            return run_context.request_stop()
        else:
            new_case = self.features_val["names"][0].decode("utf-8")
            if self.cur_case != new_case:
                # Finish a case
                self._save_guide()
                # Update states with next case
                self.cur_case = new_case
                self.bbox = self.features_val["bboxes"][0]
            else:
                # Update guide with last prediction
                self.features_val["images"][0, ..., -1] = np.maximum(
                    self.features_val["images"][0, ..., -1],
                    get_gd_image_multi_objs(predictions["TumorPred"][0, ..., 0],
                                            center_perturb=0., stddev_perturb=0.))
            self.guides.append(self.features_val["images"][0, ..., -1])

    def _save_guide(self):
        from utils import array_kits as arr_ops
        import scipy.ndimage as ndi

        img_array = np.stack(self.guides, axis=0)
        # Resize logits3d to the shape of labels3d
        ori_shape = list(arr_ops.bbox_to_shape(self.bbox))
        cur_shape = img_array.shape
        ori_shape[0] = cur_shape[0]
        scales = np.array(ori_shape) / np.array(cur_shape)
        img_array = ndi.zoom(img_array, scales, order=1)
        img_array = (img_array * 255).astype(np.int16)

        header = self.img_reader.header(self.cur_case.replace(".rev", ""))

        cur_case = Path(self.cur_case.replace(".rev", ""))
        case_name = cur_case.name.replace("volume", "guide") + ".gz"
        save_path = Path(self.model_dir) / "spatial_guide"
        if not save_path.exists():
            save_path.mkdir(parents=True, exist_ok=True)
        if self.cur_case.endswith(".rev"):
            save_path = save_path / case_name.replace("guide", "rev-guide")
            img_array = np.flip(img_array, axis=0)
        else:
            save_path = save_path / case_name

        pad_with = tuple(zip(self.bbox[2::-1], np.array(header.shape) - self.bbox[:2:-1] - 1))
        img_array = np.pad(img_array, pad_with, mode="constant", constant_values=0)

        self.img_reader.save(save_path, img_array, fmt=cur_case.suffix[1:])
        self.guides.clear()


class LogLearningRateHook(session_run_hook.SessionRunHook):
    def __init__(self,
                 tag,
                 every_n_steps=100,
                 every_n_secs=None,
                 output_dir=None,
                 summary_writer=None):
        self._timer = basic_session_run_hooks.SecondOrStepTimer(every_steps=every_n_steps,
                                                                every_secs=every_n_secs)
        self._summary_writer = summary_writer
        self._output_dir = output_dir
        self._summary_tag = "{}/learning rate".format(tag)
        self._steps_per_run = 1

    def _set_steps_per_run(self, steps_per_run):
        self._steps_per_run = steps_per_run

    def begin(self):
        if self._summary_writer is None and self._output_dir:
            self._summary_writer = SummaryWriterCache.get(self._output_dir)
        self._global_step_tensor = training_util._get_or_create_global_step_read()
        if self._global_step_tensor is None:
            raise RuntimeError(
                "Global step should be created to use StepCounterHook.")

    def after_create_session(self, session, coord):
        self._lr_tensor = ops.get_collection(CustomKeys.LEARNING_RATE)[0]

    def before_run(self, run_context):
        return session_run_hook.SessionRunArgs([self._lr_tensor, self._global_step_tensor])

    def after_run(self, run_context, run_values):
        lr, stale_global_step = run_values.results
        if self._timer.should_trigger_for_step(
                stale_global_step + self._steps_per_run):
            # get the real value after train op.
            global_step = run_context.session.run(self._global_step_tensor)
            if self._timer.should_trigger_for_step(global_step):
                self._timer.update_last_triggered_step(global_step)
                self._log_and_record(lr, global_step)

    def _log_and_record(self, lr, step):
        if self._summary_writer is not None:
            summary_scalar(self._summary_writer, step, [self._summary_tag], [lr])
        logging.info(self._summary_tag + ": {:.6f}".format(lr))
