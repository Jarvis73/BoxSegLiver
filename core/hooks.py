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
# import copy
import json
import numpy as np
# import pandas as pd
from pathlib import Path

from tensorflow.python.distribute import values
from tensorflow.python.data.ops import iterator_ops
from tensorflow.python.framework import ops
# from tensorflow.python.framework import errors_impl
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import training_util
from tensorflow.python.training import session_run_hook
from tensorflow.python.training import basic_session_run_hooks
from tensorflow.python.training import saver as saver_lib
from tensorflow.python.training.summary_io import SummaryWriterCache

# import data_kits.build_data as data_ops
from evaluators import evaluator_base
from utils.summary_kits import summary_scalar
# from utils.array_kits import get_gd_image_multi_objs
from config import CustomKeys
# from core import evaluator_liver


class IteratorStringHandleHook(session_run_hook.SessionRunHook):
    """ Hook to initialize string handle of Iterator """

    def __init__(self, train_iterator, eval_iterator):
        if isinstance(train_iterator, iterator_ops.Iterator) and \
                isinstance(eval_iterator, iterator_ops.Iterator):
            self._train_iterator = train_iterator
            self._eval_iterator = eval_iterator
        elif isinstance(train_iterator, values.PerReplicaDataIterator) and \
                isinstance(eval_iterator, values.PerReplicaDataIterator):
            self._train_iterator = train_iterator._iterator
            self._eval_iterator = eval_iterator._iterator
        else:
            raise TypeError()
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


class EvaluatorHook(session_run_hook.SessionRunHook):
    """ Hook to evaluate at epoch end during training """

    def __init__(self,
                 evaluator,
                 checkpoint_dir=None,
                 compare_fn=None,
                 prefix=None,
                 eval_n_secs=None,
                 eval_n_steps=None,
                 saver=None,
                 checkpoint_basename="best_model.ckpt",
                 save_best=False,
                 save_interval=0):
        """Initializes a `CheckpointSaverHook`.

        Args:
          prefix: `str`, summary prefix
          evaluator: for evaluate model
          checkpoint_dir: `str`, base directory for the checkpoint files.
          compare_fn: `function`, compare function for the better results
          eval_n_secs: `int`, save every N secs.
          eval_n_steps: `int`, save every N steps.
          saver: `Saver` object, used for saving.
          checkpoint_basename: `str`, base name for the checkpoint files.
          save_best: `bool`, save best ckpt or not
          save_interval: `int`: 0 for False, positive for True. It is work on if save_best is True.

        Raises:
          ValueError: One of `save_steps` or `save_secs` should be set.
          ValueError: At most one of `saver` or `scaffold` should be set.
        """
        logging.info("Create BestCheckpointSaverHook.")
        if not isinstance(evaluator, evaluator_base.EvaluateBase):
            raise TypeError("`evaluator` must be an EvaluateBase instance")
        self._summary_tag = prefix + "/Eval/{}" if prefix else "Eval/{}"
        self._evaluator = evaluator
        self._compare_fn = compare_fn
        self._saver = saver
        self._checkpoint_dir = checkpoint_dir
        self._timer = basic_session_run_hooks.SecondOrStepTimer(every_secs=eval_n_secs,
                                                                every_steps=eval_n_steps)
        self._steps_per_run = 1
        self._save_best = save_best
        self._save_interval = save_interval
        self._better_result = None

        if self._save_best:
            logging.info("Enable --> save best checkpoint!")
            self._save_path = os.path.join(checkpoint_dir, checkpoint_basename)
            self._need_save = False
            self._last_step_in_get_saver = 0
            if self._save_interval:
                logging.info("       --> save best checkpoint in each interval of %d steps!" % self._save_interval)
                saved_steps = [-1]
                for x in self._get_best_result_dump_file(name="best_result_*", use_glob=True):
                    saved_steps.append(int(x.stem.split("_")[-1]))
                max_saved_step = np.max(saved_steps)
                best_file = self._get_best_result_dump_file(name="best_result_{}".format(max_saved_step))
                self._last_step_in_get_saver = max_saved_step
            else:
                best_file = self._get_best_result_dump_file()
            if best_file.exists():
                with best_file.open() as f:
                    self._better_result = json.load(f)
                logging.info("Best result records '{}' loaded!".format(str(best_file)))

    def _set_steps_per_run(self, steps_per_run):
        self._steps_per_run = steps_per_run

    def begin(self):
        self._summary_writer = SummaryWriterCache.get(self._checkpoint_dir)
        self._global_step_tensor = training_util._get_or_create_global_step_read()  # pylint: disable=protected-access
        if self._global_step_tensor is None:
            raise RuntimeError(
                "Global step should be created to use BestCheckpointSaverHook.")

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
        results = self._evaluator.run_with_session(session)
        if self._save_interval and (step // self._save_interval !=
                                    self._last_step_in_get_saver // self._save_interval):
            # Reset self._better_result for new interval.
            self._better_result = None

        if not self._better_result or self._compare_fn(results, self._better_result):
            self._better_result = results
            self._need_save = True

        self._summary(step, results)
        if self._save_best:
            return (self._save_interval_best(session, step)
                    if self._save_interval else self._save(session, step))
        else:
            return False

    def _save(self, session, step):
        """Saves the better checkpoint, returns should_stop."""
        if not self._need_save:
            return False
        self._need_save = False
        logging.info("Saving (best) checkpoints for %d into %s (checkpoint_best).",
                     step - 1, self._save_path)

        # We must use a different latest_filename comparing with the default "checkpoint"
        self._get_saver().save(session, self._save_path, global_step=step,
                               write_meta_graph=False,
                               latest_filename="checkpoint_best")
        with self._get_best_result_dump_file().open("w") as f:
            json.dump(self._get_result_for_json_dump(), f)

        should_stop = False
        return should_stop

    def _save_interval_best(self, session, step):
        """Saves the better checkpoint in each interval, returns should_stop."""
        if not self._need_save:
            return False
        self._need_save = False

        # We must use a different latest_filename comparing with the default "checkpoint"
        end_point = (step // self._save_interval + 1) * self._save_interval
        logging.info("Saving (best) checkpoints for step %d into %s (%s).",
                     step - 1, self._save_path, "checkpoint_best_%d" % end_point)
        self._get_saver(step).save(session, self._save_path, global_step=step,
                                   write_meta_graph=False,
                                   latest_filename="checkpoint_best_{}".format(end_point))
        with self._get_best_result_dump_file("best_result_{}".format(end_point)).open("w") as f:
            json.dump(self._get_result_for_json_dump(), f)

        should_stop = False
        return should_stop

    def _summary(self, step, result=None):
        if result is None:
            result = self._better_result

        tags, values_ = [], []
        for key, value in result.items():
            if key == ops.GraphKeys.GLOBAL_STEP:
                continue
            tags.append(self._summary_tag.format(key))
            values_.append(value)

        summary_scalar(self._summary_writer, step, tags, values_)

    def _get_result_for_json_dump(self):
        res = {}
        for key, val in self._better_result.items():
            if isinstance(val, np.int64):
                val = int(val)
            else:
                val = float(val)
            res[key] = val

        return res

    def _get_best_result_dump_file(self, name="best_result", use_glob=False):
        if use_glob:
            return Path(self._save_path).parent.glob(name)
        return Path(self._save_path).parent / name

    def _get_saver(self, step=None):
        if self._saver is not None and (
                step is None or (step // self._save_interval ==
                                 self._last_step_in_get_saver // self._save_interval)):
            self._last_step_in_get_saver = step
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
        self._last_step_in_get_saver = step
        return self._saver


class EvaluatorHookV2(session_run_hook.SessionRunHook):
    """ Hook to evaluate at epoch end during training
    V2 save best checkpoint by the metric with moving average
    """

    def __init__(self,
                 evaluator,
                 checkpoint_dir=None,
                 compare_fn=lambda x, y: x > y,
                 prefix=None,
                 eval_n_secs=None,
                 eval_n_steps=None,
                 saver=None,
                 checkpoint_basename="best_model.ckpt",
                 save_best=False,
                 ma_alpha=0.9):
        """Initializes a `CheckpointSaverHook`.

        Args:
          evaluator: for evaluate model
          checkpoint_dir: `str`, base directory for the checkpoint files.
          compare_fn: `function`, compare function for the better results
          prefix: `str`, summary prefix
          eval_n_secs: `int`, save every N secs.
          eval_n_steps: `int`, save every N steps.
          saver: `Saver` object, used for saving.
          checkpoint_basename: `str`, base name for the checkpoint files.
          save_best: `bool`, save best ckpt or not

        Raises:
          ValueError: One of `save_steps` or `save_secs` should be set.
          ValueError: At most one of `saver` or `scaffold` should be set.
        """
        logging.info("Create BestCheckpointSaverHook V2(MA).")
        if not isinstance(evaluator, evaluator_base.EvaluateBase):
            raise TypeError("`evaluator` must be an EvaluateBase instance")
        self._summary_tag = prefix + "/Eval/{}" if prefix else "Eval/{}"
        self._evaluator = evaluator
        self._compare_fn = compare_fn
        self._saver = saver
        self._checkpoint_dir = checkpoint_dir
        self._timer = basic_session_run_hooks.SecondOrStepTimer(every_secs=eval_n_secs,
                                                                every_steps=eval_n_steps)
        self._steps_per_run = 1
        self._save_best = save_best
        self._ma_results = None
        self._ma_best_result = None
        self.ma_alpha = ma_alpha
        self._trigger_counter = 0

        if self._save_best:
            logging.info("Enable --> save best checkpoint!")
            self._save_path = os.path.join(checkpoint_dir, checkpoint_basename)
            self._need_save = False
            self._last_step_in_get_saver = 0
            best_file = self._get_best_result_dump_file()
            if best_file.exists():
                with best_file.open() as f:
                    data = json.load(f)
                    self._ma_results = data["ma_results"]
                    self._ma_best_result = data["ma_best_result"]
                logging.info("Load previous best result records: ", data)

    def _set_steps_per_run(self, steps_per_run):
        self._steps_per_run = steps_per_run

    def begin(self):
        self._summary_writer = SummaryWriterCache.get(self._checkpoint_dir)
        self._global_step_tensor = training_util._get_or_create_global_step_read()  # pylint: disable=protected-access
        if self._global_step_tensor is None:
            raise RuntimeError(
                "Global step should be created to use BestCheckpointSaverHook.")

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
                self._trigger_counter += 1
                if self._evaluate(run_context.session, global_step):
                    run_context.request_stop()

    def end(self, session):
        last_step = session.run(self._global_step_tensor)
        if last_step != self._timer.last_triggered_step():
            self._evaluate(session, last_step)

    def _evaluate(self, session, step):
        results = self._evaluator.run_with_session(session)
        # We update moving average for 1 trigger delay
        if self._trigger_counter <= 1:
            return False
        if self._ma_results is None:
            self._ma_results = {k: float(v) for k, v in results.items()}
            self._ma_best_result = np.mean(list(results.values()))
            self._need_save = True
        else:
            self._ma_results = {k: float(self.ma_alpha * v + (1 - self.ma_alpha) * results[k])
                                for k, v in self._ma_results.items()}
            new_avg = np.mean(list(self._ma_results.values()))
            if self._compare_fn(new_avg, self._ma_best_result):
                self._ma_best_result = new_avg
                self._need_save = True

        self._summary(step, self._ma_results)
        if self._save_best and self._need_save:
            self._need_save = False
            return self._save(session, step)
        else:
            return False

    def _save(self, session, step):
        """Saves the better checkpoint, returns should_stop."""
        logging.info("Saving (best) checkpoints for %d into %s (checkpoint_best).",
                     step - 1, self._save_path)

        # We must use a different latest_filename comparing with the default "checkpoint"
        self._get_saver().save(session, self._save_path, global_step=step,
                               write_meta_graph=False,
                               latest_filename="checkpoint_best")
        with self._get_best_result_dump_file().open("w") as f:
            json.dump(self._get_result_for_json_dump(), f)

        should_stop = False
        return should_stop

    def _summary(self, step, result=None):
        if result is None:
            result = self._ma_results

        tags, values_ = [], []
        for key, value in result.items():
            if key == ops.GraphKeys.GLOBAL_STEP:
                continue
            tags.append(self._summary_tag.format(key))
            values_.append(value)

        summary_scalar(self._summary_writer, step, tags, values_)

    def _get_result_for_json_dump(self):
        res = {"ma_results": self._ma_results, "ma_best_result": float(self._ma_best_result)}
        return res

    def _get_best_result_dump_file(self, name="best_result", use_glob=False):
        if use_glob:
            return Path(self._save_path).parent.glob(name)
        return Path(self._save_path).parent / name

    def _get_saver(self, step=None):
        if self._saver is not None and (
                step is None or (step // self._save_interval ==
                                 self._last_step_in_get_saver // self._save_interval)):
            self._last_step_in_get_saver = step
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
        self._last_step_in_get_saver = step
        return self._saver


# class FeedGuideHook(session_run_hook.SessionRunHook):
#     def __init__(self, features_ph, labels_ph, features, labels, model_dir, model_args):
#         self.args = model_args
#         self.features_ph = copy.copy(features_ph)    # Copy a new dict
#         self.labels_ph = labels_ph
#         self.features = features
#         self.labels = labels
#         self.features_val = None
#         self.labels_val = None
#         self.predictions = None
#         self.first = True
#         self.guides = []
#         self.cur_case = None
#         self.bbox = None
#         self.img_reader = data_ops.ImageReader()
#         self.model_dir = model_dir
#         self.cbn = False
#         self.id = 0     # ID(indices) of slices in a CT volume
#
#         # load interactive information: z_min, z_max
#         tumor_path = Path(__file__).parent / "data/LiTS/tumor_summary.csv"
#         tumors_info = pd.read_csv(str(tumor_path))
#         self.t_mgr = TumorManager(tumors_info, min_std=self.args.min_std)
#
#     def before_run(self, run_context):
#         if self.first:
#             self.features_val, self.labels_val = run_context.session.run(
#                 [self.features, self.labels])
#             self.first = False
#             self.cur_case = self.features_val["names"][0].decode("utf-8")
#             self.bbox = self.features_val["bboxes"][0]
#             if "sp_guide" in self.features_val:
#                 self.cbn = True
#                 # self.guides.append(self.features_val["sp_guide"][0, ..., 0])
#                 self.t_mgr.name = self.cur_case
#                 self.t_mgr.set_bbox(self.bbox, shape=self.features_val["images"].shape[1:-1])
#                 # Change features_val["sp_guide"] from moments to image
#                 self.features_val["sp_guide"] = self.t_mgr.get_guide_image(
#                     self.features_val["sp_guide"][0], new_id=self.id)
#                 self.id += 1
#                 self.guides.append(self.features_val["sp_guide"][0, ..., 0])
#             else:
#                 self.guides.append(self.features_val["images"][0, ..., -1])
#
#         feed_dict = {value: self.features_val[key] for key, value in self.features_ph.items()}
#         feed_dict[self.labels_ph] = self.labels_val
#
#         return session_run_hook.SessionRunArgs(self.predictions, feed_dict=feed_dict)
#
#     def after_run(self, run_context, run_values):
#         predictions = run_values.results
#
#         try:
#             self.features_val, self.labels_val = run_context.session.run(
#                 [self.features, self.labels])
#         except errors_impl.OutOfRangeError:
#             self._save_guide()
#             self.guides.clear()
#             return run_context.request_stop()
#         else:
#             new_case = self.features_val["names"][0].decode("utf-8")
#             if self.cur_case != new_case:
#                 # Finish a case
#                 self._save_guide()
#                 self.guides.clear()
#                 # Update states with next case
#                 self.cur_case = new_case
#                 self.bbox = self.features_val["bboxes"][0]
#                 # Reinitialize TumorManager
#                 need_rev = self.cur_case.endswith("rev")
#                 self.id = self.id - 1 if need_rev else 0
#                 self.t_mgr.reset(direction=-1 if need_rev else 1)
#                 self.t_mgr.name = self.cur_case
#                 self.t_mgr.set_bbox(self.bbox, shape=self.features_val["images"].shape[1:-1])
#                 self.features_val["sp_guide"] = self.t_mgr.get_guide_image(
#                     self.features_val["sp_guide"][0], new_id=self.id)
#                 if self.t_mgr.direction == 1:
#                     self.id += 1
#                 else:
#                     self.id -= 1
#             else:
#                 # Update guide with last prediction
#                 self.t_mgr.check_pred(predictions["TumorPred"][0, ..., 0])
#                 if self.cbn:
#                     # self.features_val["sp_guide"][0, ..., 0] = np.maximum(
#                     #     self.features_val["sp_guide"][0, ..., 0],
#                     #     get_gd_image_multi_objs(corrective_tumor,
#                     #                             center_perturb=0., stddev_perturb=0.))
#                     self.features_val["sp_guide"] = self.t_mgr.get_guide_image(
#                         self.features_val["sp_guide"][0], self.id)
#                     if self.t_mgr.direction == 1:
#                         self.id += 1
#                     else:
#                         self.id -= 1
#                 else:
#                     self.features_val["images"][0, ..., -1] = np.maximum(
#                         self.features_val["images"][0, ..., -1],
#                         get_gd_image_multi_objs(self.t_mgr.pred,
#                                                 center_perturb=0., stddev_perturb=0.))
#             if self.cbn:
#                 self.guides.append(self.features_val["sp_guide"][0, ..., 0])
#             else:
#                 self.guides.append(self.features_val["images"][0, ..., -1])
#
#     def _save_guide(self):
#         from utils import array_kits as arr_ops
#         import scipy.ndimage as ndi
#
#         img_array = np.stack(self.guides, axis=0)
#         # Resize logits3d to the shape of labels3d
#         ori_shape = list(arr_ops.bbox_to_shape(self.bbox))
#         cur_shape = img_array.shape
#         ori_shape[0] = cur_shape[0]
#         scales = np.array(ori_shape) / np.array(cur_shape)
#         img_array = ndi.zoom(img_array, scales, order=1)
#         img_array = (img_array * 255).astype(np.int16)
#
#         header = self.img_reader.header(self.cur_case.replace(".rev", ""))
#
#         cur_case = Path(self.cur_case.replace(".rev", ""))
#         case_name = cur_case.name.replace("volume", "guide") + ".gz"
#         save_path = Path(self.model_dir) / "spatial_guide"
#         if not save_path.exists():
#             save_path.mkdir(parents=True, exist_ok=True)
#         if self.cur_case.endswith(".rev"):
#             save_path = save_path / case_name.replace("guide", "rev-guide")
#             img_array = np.flip(img_array, axis=0)
#         else:
#             save_path = save_path / case_name
#
#         pad_with = tuple(zip(self.bbox[2::-1], np.array(header.shape) - self.bbox[:2:-1] - 1))
#         img_array = np.pad(img_array, pad_with, mode="constant", constant_values=0)
#
#         self.img_reader.save(save_path, img_array, fmt=cur_case.suffix[1:])


class LogLearningRateHook(session_run_hook.SessionRunHook):
    def __init__(self,
                 prefix,
                 every_n_steps=100,
                 every_n_secs=None,
                 output_dir=None,
                 do_logging=True,
                 summary_writer=None):
        self._timer = basic_session_run_hooks.SecondOrStepTimer(every_steps=every_n_steps,
                                                                every_secs=every_n_secs)
        self._summary_writer = summary_writer
        self._output_dir = output_dir
        self._summary_tag = "{}/learning rate".format(prefix)
        self._steps_per_run = 1
        self.do_logging = do_logging

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
        if self.do_logging:
            logging.info(self._summary_tag + ": {:.6f}".format(lr))


class LoggingTensorWithSpeedFormatterHook(basic_session_run_hooks.LoggingTensorHook):
    """ Modified LoggingTensorHook with speed and custom formatter
    """

    @staticmethod
    def custom_formatter(tensor_values, round_digits=3):
        stats = []
        _tag_order = sorted(tensor_values.keys())
        for tag in _tag_order:
            if tag == "step":
                stats.append("%s = %.{}g".format(4) % (tag, tensor_values[tag]))
            else:
                stats.append("%s = %.{}g".format(round_digits) % (tag, tensor_values[tag]))
        return ', '.join(stats)

    def _log_tensors(self, tensor_values):
        last_step = self._timer.last_triggered_step()
        elapsed_secs, _ = self._timer.update_last_triggered_step(self._iter_count)
        if elapsed_secs is not None:
            speed = (self._iter_count - last_step) / elapsed_secs
            logging.info(self._formatter(tensor_values) + " ({:.3g} it/s)".format(speed))
        else:
            logging.info(self._formatter(tensor_values))


class AverageTensorHook(session_run_hook.SessionRunHook):
    def __init__(self,
                 update_op,
                 local_init_ops,
                 every_n_steps=200,
                 every_n_secs=None):
        self.update_op = update_op
        self.local_init_ops = local_init_ops
        self._timer = basic_session_run_hooks.SecondOrStepTimer(every_steps=every_n_steps,
                                                                every_secs=every_n_secs)
        self._steps_per_run = 1

    def _set_steps_per_run(self, steps_per_run):
        self._steps_per_run = steps_per_run

    def begin(self):
        self._global_step_tensor = training_util._get_or_create_global_step_read()
        if self._global_step_tensor is None:
            raise RuntimeError(
                "Global step should be created to use StepCounterHook.")

    def before_run(self, run_context):
        return session_run_hook.SessionRunArgs([self.update_op, self._global_step_tensor])

    def after_run(self, run_context, run_values):
        lr, stale_global_step = run_values.results
        if self._timer.should_trigger_for_step(
                stale_global_step + self._steps_per_run):
            # get the real value after train op.
            global_step = run_context.session.run(self._global_step_tensor)
            if self._timer.should_trigger_for_step(global_step):
                self._timer.update_last_triggered_step(global_step)
                run_context.session.run(self.local_init_ops)


class ReduceLROnPlateauHook(session_run_hook.SessionRunHook):
    def __init__(self,
                 save_dir,
                 monitor='total_loss',
                 lr_patience=30,
                 tr_patience=50,
                 mode='min',
                 min_delta=0.0005,
                 cooldown=0,
                 moving_average=0.95,
                 every_n_steps=200,
                 every_n_secs=None):
        self.save_dir = save_dir
        self.monitor = monitor
        self.lr_patience = lr_patience    # wait number of epochs
        self.tr_patience = tr_patience  # wait number of epochs
        self.mode = mode
        self.min_delta = min_delta
        self.cooldown = cooldown
        self.cooldown_counter = 0  # Cooldown counter.
        self.lr_wait = 0
        self.tr_wait = 0
        self.best = 0
        self.monitor_op = None
        self.alpha = moving_average
        self.total_loss_MA = None
        self.lr = None
        self.lr_threshold = 1e-6
        self._reset()
        self.load_lr_schedule()
        self.inc_tr_patience = self.tr_patience // 2
        self._timer = basic_session_run_hooks.SecondOrStepTimer(every_steps=every_n_steps,
                                                                every_secs=every_n_secs)
        self._steps_per_run = 1

    def _set_steps_per_run(self, steps_per_run):
        self._steps_per_run = steps_per_run

    def begin(self):
        self._global_step_tensor = training_util._get_or_create_global_step_read()
        if self._global_step_tensor is None:
            raise RuntimeError(
                "Global step should be created to use StepCounterHook.")

    def after_create_session(self, session, coord):
        if self.monitor == "total_loss":
            self.monitor = ops.get_collection(CustomKeys.LOSS_MEAN)[0]
        self.lr = ops.get_collection(CustomKeys.LEARNING_RATE)[0]
        self.update_op = ops.get_collection(CustomKeys.LR_UPDATE_OPS)[0]

    def before_run(self, run_context):
        return session_run_hook.SessionRunArgs([self.lr, self.monitor, self._global_step_tensor])

    def after_run(self, run_context, run_values):
        old_lr, current, stale_global_step = run_values.results
        if self._timer.should_trigger_for_step(
                stale_global_step + self._steps_per_run):
            # get the real value after train op.
            global_step = run_context.session.run(self._global_step_tensor)
            if self._timer.should_trigger_for_step(global_step) and global_step > 2:
                self._timer.update_last_triggered_step(global_step)
                self.try_update_lr(run_context.session, current)
                if self.check_stop(old_lr):
                    run_context.request_stop()

    def load_lr_schedule(self):
        schedule_file = Path(self.save_dir) / "lr_schedule"
        if schedule_file.exists():
            with schedule_file.open() as f:
                lr_schedule = json.load(f)
            self.best = lr_schedule["best"]
            # self.lr_patience = lr_schedule["lr_patience"]
            # self.tr_patience = lr_schedule["tr_patience"]
            self.total_loss_MA = lr_schedule["total_loss_MA"]
            self.tr_wait = lr_schedule["tr_wait"]
            self.lr_wait = lr_schedule["lr_wait"]
            self.cooldown_counter = lr_schedule["cooldown_counter"]

    def save_lr_schedule(self):
        schedule_file = Path(self.save_dir) / "lr_schedule"
        lr_schedule = {"best": float(self.best),
                       "total_loss_MA": float(self.total_loss_MA),
                       "tr_wait": self.tr_wait,
                       "lr_wait": self.lr_wait,
                       "lr_patience": self.lr_patience,
                       "lr_threshold": float(self.lr_threshold),
                       "tr_patience": self.tr_patience,
                       "cooldown_counter": self.cooldown_counter,
                       "mode": self.mode}
        with schedule_file.open("w") as f:
            json.dump(lr_schedule, f)

    def try_update_lr(self, session, current):
        if self.total_loss_MA is None:
            self.total_loss_MA = current
        else:
            self.total_loss_MA = self.alpha * self.total_loss_MA + (1 - self.alpha) * current

        if self.in_cooldown():
            self.cooldown_counter -= 1
            self.lr_wait = 0

        logging.info("*** total_loss_MA={:.3g}, last_best={:.3g}, wait {} epochs/tr, {} epochs/lr"
                     .format(self.total_loss_MA, self.best, self.tr_wait, self.lr_wait))
        if self.monitor_op(self.total_loss_MA, self.best):
            self.best = self.total_loss_MA
            self.lr_wait = 0
            self.tr_wait = 0
        elif not self.in_cooldown():
            self.lr_wait += 1
            self.tr_wait += 1
            if self.lr_wait > self.lr_patience:
                logging.info("*** Decay learning rate. Total loss MA: {:.3g}".format(self.total_loss_MA))
                session.run(self.update_op)
                self.cooldown_counter = self.cooldown
                self.lr_wait = 0

        self.save_lr_schedule()

    def check_stop(self, old_lr):
        if self.tr_wait <= self.tr_patience:
            return False
        elif old_lr > self.lr_threshold:
            self.tr_wait -= self.inc_tr_patience
            return False
        return True

    def _reset(self):
        """Resets wait counter and cooldown counter."""
        if self.mode not in ['min', 'max']:
            raise ValueError('Learning Rate Plateau Reducing mode %s is unknown, '
                             'fallback to auto mode.', self.mode)
        if self.mode == 'min':
            self.monitor_op = lambda a, b: np.less(a, b - self.min_delta)
            self.best = np.Inf
        else:
            self.monitor_op = lambda a, b: np.greater(a, b + self.min_delta)
            self.best = -np.Inf
        self.cooldown_counter = 0
        self.wait = 0

    def in_cooldown(self):
        return self.cooldown_counter > 0
