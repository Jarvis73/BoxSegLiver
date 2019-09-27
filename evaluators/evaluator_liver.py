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

import json
import numpy as np
import nibabel as nib
import tensorflow as tf
import tensorflow_estimator as tfes
import scipy.ndimage as ndi
from collections import defaultdict
from pathlib import Path

import loss_metrics as metric_ops
import utils.array_kits as arr_ops
from evaluators.evaluator_base import EvaluateBase
from utils import timer
from DataLoader.Liver import nii_kits
from DataLoader.Liver import input_pipeline_g

ModeKeys = tfes.estimator.ModeKeys


def add_arguments(parser):
    group = parser.add_argument_group(title="Evaluation Arguments")
    group.add_argument("--primary_metric",
                       type=str,
                       required=False, help="Primary metric for evaluation. Typically it has format "
                                            "<class>/<metric>")
    group.add_argument("--secondary_metric",
                       type=str,
                       required=False, help="Secondary metric for evaluation. Typically it has format "
                                            "<class>/<metric>")
    group.add_argument("--eval_final",
                       action="store_true",
                       required=False, help="Evaluate with final checkpoint. If not set, then evaluate "
                                            "with best checkpoint(default).")
    group.add_argument("--ckpt_path",
                       type=str,
                       required=False, help="Given a specified checkpoint for evaluation. "
                                            "(default best checkpoint)")
    group.add_argument("--evaluator",
                       type=str,
                       choices=["Volume"])
    group.add_argument("--eval_num",
                       type=int, default=-1,
                       required=False, help="Number of cases for evaluation")
    group.add_argument("--eval_skip_num",
                       type=int,
                       default=0,
                       required=False, help="Skip some cases for evaluating determined case")
    group.add_argument("--eval_3d",
                       action="store_true",
                       required=False, help="Evaluate in 2D slices or 3D volume when training."
                                            "Default in 2D slices")
    group.add_argument("--pred_type", type=str, choices=["pred", "prob"],
                       default="pred", help="Generate prediction or probability")
    group.add_argument("--save_path", type=str, default="prediction")
    group.add_argument("--use_global_dice", action="store_true")


def get_evaluator(evaluator, estimator=None, model_dir=None, params=None,
                  merge_tumor_to_liver=True, largest=True, use_sg_reduce_fp=False):
    if evaluator == "Volume":
        return EvaluateVolume(estimator,
                              model_dir=model_dir,
                              params=params,
                              merge_tumor_to_liver=merge_tumor_to_liver,
                              largest=largest,
                              use_sg_reduce_fp=use_sg_reduce_fp)
    # elif evaluator == "Slice":
    #     return EvaluateSlice(estimator)
    else:
        raise ValueError("Unsupported evaluator: {}. Must be [Volume, ]".format(evaluator))


class EvaluateVolume(EvaluateBase):
    """ Evaluate Estimator model by volume

    This class is for liver and tumor segmentation.

    `estimator` used in run_with_session()
    `model_dir` used in run()
    """
    def __init__(self, estimator=None, model_dir=None, params=None,
                 merge_tumor_to_liver=True, largest=True, use_sg_reduce_fp=False):
        self.estimator = estimator
        self.model_dir = model_dir or estimator.model_dir
        self.params = params or estimator.params
        self.config = self.params["args"]
        self._timer = timer.Timer()
        meta_file = Path(__file__).parent.parent / "DataLoader/Liver/prepare/meta.json"
        with meta_file.open() as f:
            meta = json.load(f)
        self.meta = {x["PID"]: x for x in meta}
        if hasattr(self.config, "eval_in_patches"):
            self.eval_in_patches = self.config.eval_in_patches
        else:
            self.eval_in_patches = False
        self.do_mirror = self.config.eval_mirror
        if self.do_mirror:
            if self.config.random_flip in [1, 2]:
                self.mirror_div = 2
            elif self.config.random_flip == 3:
                self.mirror_div = 4
            else:
                self.mirror_div = 1
            tf.logging.info("Enable --> average by mirror, divisor = {}".format(self.mirror_div))
        else:
            self.mirror_div = 1

        self.merge_tumor_to_liver = merge_tumor_to_liver
        self.largest = largest
        if self.merge_tumor_to_liver:
            tf.logging.info("Enable --> merge_tumor_to_liver")
        if self.largest:
            tf.logging.info("Enable --> largest")
        if hasattr(self.config, "use_spatial") and self.config.use_spatial:
            tf.logging.info("Enable --> use spatial guide")
            self.use_sg_reduce_fp = use_sg_reduce_fp
            if self.use_sg_reduce_fp:
                tf.logging.info("Enable --> use_sg_reduce_fp")
        else:
            self.use_sg_reduce_fp = False
        if hasattr(self.config, "use_context") and self.config.use_context:
            tf.logging.info("Enable --> use context guide")

    @property
    def classes(self):
        return self.params["model_instances"][0].classes[1:]  # Remove background

    @property
    def metrics_str(self):
        return self.config.metrics_eval

    @staticmethod
    def maybe_append(dst, src, name, clear=False):
        if clear:
            dst.clear()
        if name in src:
            dst.append(src[name])

    def find_checkpoint_path(self, checkpoint_dir, latest_filename):
        if not checkpoint_dir:
            latest_path = tf.train.latest_checkpoint(self.model_dir, latest_filename)
            if not latest_path:
                tf.logging.info('Could not find trained model in model_dir: {}, running '
                                'initialization to evaluate.'.format(self.model_dir))
            checkpoint_dir = latest_path
        return checkpoint_dir

    def run_with_session(self, session=None):
        # TODO(zjw) Maybe add eval with mirror here
        if self.config.eval_3d:
            return self._run_with_session_actual_3d(session)
        else:
            return self._run_with_session_actual_2d(session)

    def _run_with_session_actual_3d(self, session=None):
        data_list = input_pipeline_g._collect_datasets(
            self.config.test_fold, "eval_online",
            filter_tumor_size=self.config.filter_size,
            filter_only_liver_in_val=self.params.get("filter_only_liver_in_val", True))
        data_dict = {str(case["PID"]): case for case in data_list}
        # We just use model_instances[0] to get prediction name.
        # Its value will still be collected from multi-gpus
        predicts = ["labels", "names"] + list(self.params["model_instances"][0].predictions)
        predict_gen = self.estimator.evaluate_online(session, predicts, yield_single_examples=False)
        cur_case = None
        self._timer.reset()
        self.clear_metrics()

        if not self.config.use_global_dice:
            tf.logging.info("Begin evaluating 3d at epoch end ...")
            self._timer.tic()
            volume_collection = defaultdict(list)
            for x in predict_gen:
                new_case = str(x["names"][0])
                cur_case = cur_case or new_case
                if cur_case == new_case:    # Append batch to collections
                    for cls in self.classes:
                        volume_collection[cls].append(np.squeeze(x[cls + "Pred"], axis=-1))
                    volume_collection["labels"].append(x["labels"])
                else:
                    volume_collection = {k: np.concatenate(v, axis=0) for k, v in volume_collection.items()}
                    z1, y1, x1, z2, y2, x2 = data_dict[cur_case]["bbox"]
                    pads = (self.config.batch_size - ((z2 - z1) % self.config.batch_size)) % self.config.batch_size
                    if pads > 0:
                        volume_collection = {k: v[:-pads] for k, v in volume_collection.items()}
                    volume = volume_collection
                    labels = volume_collection["labels"]
                    # Here we omit postprocess for saving training time
                    # volume = self._postprocess(volume, ori_shape=[y2 - y1, x2 - x1])
                    # labels = self._postprocess(labels, is_label=True, ori_shape=[y2 - y1, x2 - x1])
                    results = {}
                    for c, cls in enumerate(self.classes):
                        pairs = metric_ops.metric_3d(volume[cls], labels[cls], required=self.metrics_str)
                        for met, value in pairs.items():
                            results["{}/{}".format(cls, met)] = value
                    self.append_metrics(results)
                    volume_collection.clear()
                    for cls in self.classes:
                        volume_collection[cls].append(np.squeeze(x[cls + "Pred"], axis=-1))
                    volume_collection["labels"].append(x["labels"])
                    cur_case = new_case
                self._timer.toc()
                self._timer.tic()

            volume_collection = {k: np.concatenate(v, axis=0) for k, v in volume_collection.items()}
            z1, y1, x1, z2, y2, x2 = data_dict[cur_case]["bbox"]
            pads = (self.config.batch_size - ((z2 - z1) % self.config.batch_size)) % self.config.batch_size
            if pads > 0:
                volume_collection = {k: v[:-pads] for k, v in volume_collection.items()}
            volume = volume_collection
            labels = volume["labels"]
            results = {}
            for c, cls in enumerate(self.classes):
                pairs = metric_ops.metric_3d(volume[cls], labels[cls], required=self.metrics_str)
                for met, value in pairs.items():
                    results["{}/{}".format(cls, met)] = value
            self.append_metrics(results)

            display_str = "----Evaluate {} batches ".format(self._timer.calls)
            results = {key: np.mean(values) for key, values in self._metric_values.items()}
        else:
            tf.logging.info("Begin evaluating 3d at epoch end(global dice) ...")
            accumulator = defaultdict(int)
            self._timer.tic()
            volume_collection = defaultdict(list)
            for x in predict_gen:
                new_case = str(x["names"][0])
                cur_case = cur_case or new_case
                if cur_case == new_case:  # Append batch to collections
                    for cls in self.classes:
                        volume_collection[cls].append(np.squeeze(x[cls + "Pred"], axis=-1))
                    volume_collection["labels"].append(x["labels"])
                else:
                    volume_collection = {k: np.concatenate(v, axis=0) for k, v in volume_collection.items()}
                    z1, y1, x1, z2, y2, x2 = data_dict[cur_case]["bbox"]
                    pads = (self.config.batch_size - ((z2 - z1) % self.config.batch_size)) % self.config.batch_size
                    if pads > 0:
                        volume_collection = {k: v[:-pads] for k, v in volume_collection.items()}
                    volume = volume_collection
                    labels = volume_collection["labels"]
                    # Here we omit postprocess for saving training time
                    # volume = self._postprocess(volume_collection, ori_shape=[y2 - y1, x2 - x1])
                    # labels = self._postprocess(volume_collection["labels"], is_label=True,
                    #                            ori_shape=[y2 - y1, x2 - x1])
                    for i, cls in enumerate(self.classes):
                        conf = metric_ops.ConfusionMatrix(volume[cls].astype(int), (labels == i + 1).astype(int))
                        conf.compute()
                        accumulator[cls + "_fn"] += conf.fn
                        accumulator[cls + "_fp"] += conf.fp
                        accumulator[cls + "_tp"] += conf.tp
                    volume_collection = defaultdict(list)
                    for cls in self.classes:
                        volume_collection[cls].append(np.squeeze(x[cls + "Pred"], axis=-1))
                    volume_collection["labels"].append(x["labels"])
                    cur_case = new_case

                self._timer.toc()
                self._timer.tic()

            display_str = "----Evaluate {} batches ".format(self._timer.calls)
            results = {cls + "/Dice": 2 * accumulator[cls + "_tp"] / (
                    2 * accumulator[cls + "_tp"] + accumulator[cls + "_fn"] + accumulator[cls + "_fp"])
                       for cls in self.classes}

        for key, value in results.items():
            display_str += "- {}: {:.3f} ".format(key, value)
        tf.logging.info(display_str + "({:.3f} secs)".format(self._timer.total_time))
        return results

    def _run_with_session_actual_2d(self, session=None):
        if not self.config.use_global_dice:
            accumulator = defaultdict(list)
            predicts = list(self.params["model_instances"][0].metrics_dict)
            tf.logging.info("Begin evaluating 2d at epoch end ...")
            predict_gen = self.estimator.evaluate_online(session, predicts, yield_single_examples=False)

            self._timer.reset()
            self._timer.tic()
            for x in predict_gen:
                self._timer.toc()
                for k, v in x.items():
                    accumulator[k].append(v)
                self._timer.tic()

            display_str = "----Evaluate {} batches ".format(self._timer.calls)
            results = {key: np.mean(values) for key, values in accumulator.items()}
        else:
            accumulator = defaultdict(int)
            # We just use model_instances[0] to get prediction name.
            # Its value will still be collected from multi-gpus
            predicts = ["labels"] + list(self.params["model_instances"][0].predictions)
            tf.logging.info("Begin evaluating 2d patches at epoch end(global dice) ...")
            predict_gen = self.estimator.evaluate_online(session, predicts, yield_single_examples=False)

            self._timer.reset()
            self._timer.tic()
            for x in predict_gen:
                for i, cls in enumerate(self.classes):
                    conf = metric_ops.ConfusionMatrix(np.squeeze(x[cls + "Pred"], axis=-1).astype(int),
                                                      (x["labels"] == i + 1).astype(int))
                    conf.compute()
                    accumulator[cls + "_fn"] += conf.fn
                    accumulator[cls + "_fp"] += conf.fp
                    accumulator[cls + "_tp"] += conf.tp
                self._timer.toc()
                self._timer.tic()

            display_str = "----Evaluate {} batches ".format(self._timer.calls)
            results = {cls + "/Dice": 2 * accumulator[cls + "_tp"] / (
                    2 * accumulator[cls + "_tp"] + accumulator[cls + "_fn"] + accumulator[cls + "_fp"])
                       for cls in self.classes}

        for key, value in results.items():
            display_str += "- {}: {:.3f} ".format(key, value)
        tf.logging.info(display_str + "({:.3f} secs)".format(self._timer.total_time))
        return results

    # def _evaluate_images(self, predicts, cases=None, verbose=0, save=False, do_eval=True):
    #     # TODO(zjw): should merge to _evaluate_images_do_mirror()
    #     # process a 3D image slice by slice
    #     logits3d = defaultdict(list)
    #     labels3d = defaultdict(list)
    #     # bg_masks3d = list()
    #     self.clear_metrics()
    #
    #     pad = -1
    #     bbox = None
    #     cur_case = None
    #     reshape_ori = None
    #
    #     self._timer.reset()
    #     self._timer.tic()
    #     for predict in predicts:
    #         new_case = str(predict["names"])
    #
    #         cur_case = cur_case or new_case
    #         pad = pad if pad != -1 else predict["pads"]
    #         if "bboxes" in predict:
    #             bbox = bbox if bbox is not None else predict["bboxes"]
    #         if "reshape_ori" in predict and reshape_ori is None:
    #             reshape_ori = predict["reshape_ori"]
    #
    #         if cur_case == new_case:
    #             # Append batch to collections
    #             if not predict["mirror"]:
    #                 for c, cls in enumerate(self.classes):
    #                     logits3d[cls].append(np.squeeze(predict[cls + "Pred"], axis=-1))
    #                     labels3d[cls].append(predict["labels"] == c + 1)
    #             else:
    #                 for c, cls in enumerate(self.classes):
    #                     logits3d[cls][-1] = (logits3d[cls][-1] +
    #                                          np.flip(np.squeeze(predict[cls + "Pred"], axis=-1), axis=2)) / 2
    #         else:
    #             result = self._evaluate_case(logits3d, labels3d, cur_case, pad, bbox, save,
    #                                          True, reshape_ori, do_eval)
    #             self._timer.toc()
    #             if verbose:
    #                 log_str = ("Evaluate-{} {}".format(self._timer.calls, cur_case)
    #                            if results else "Predict-{} {}".format(self._timer.calls, cur_case))
    #                 for key, value in result.items():
    #                     log_str += " {}: {:.3f}".format(key, value)
    #                 if verbose == 1:
    #                     tf.logging.debug(log_str + " ({:.3f} s)".format(self._timer.diff))
    #                 elif verbose == 2:
    #                     tf.logging.info(log_str + " ({:.3f} s)".format(self._timer.diff))
    #
    #             for c, cls in enumerate(self.classes):
    #                 logits3d[cls].clear()
    #                 labels3d[cls].clear()
    #                 logits3d[cls].append(np.squeeze(predict[cls + "Pred"], axis=-1))
    #                 labels3d[cls].append(predict["labels"] == c + 1)
    #
    #             if cases is not None and self._timer.calls >= cases:
    #                 break
    #
    #             # Reset
    #             cur_case = new_case
    #             pad = predict["pads"]
    #             if "bboxes" in predict:
    #                 bbox = predict["bboxes"]
    #             if "reshape_ori" in predict:
    #                 reshape_ori = predict["reshape_ori"]
    #             self._timer.tic()
    #
    #     if cases is None or (self._timer.calls < cases):
    #         # Final case
    #         result = self._evaluate_case(logits3d, labels3d, cur_case, pad, bbox, save,
    #                                      True, reshape_ori, do_eval)
    #         self._timer.toc()
    #         if verbose:
    #             log_str = ("Evaluate-{} {}".format(self._timer.calls, cur_case)
    #                        if results else "Predict-{} {}".format(self._timer.calls, cur_case))
    #             for key, value in result.items():
    #                 log_str += " {}: {:.3f}".format(key, value)
    #             if verbose == 1:
    #                 tf.logging.debug(log_str + " ({:.3f} s)".format(self._timer.diff))
    #             else:
    #                 tf.logging.info(log_str + " ({:.3f} s)".format(self._timer.diff))
    #
    #     # Compute average metrics
    #     display_str = "----Process {} cases ".format(self._timer.calls)
    #     results = {key: np.mean(values) for key, values in self._metric_values.items()}
    #     for key, value in results.items():
    #         display_str += "- {}: {:.3f} ".format(key, value)
    #     tf.logging.info(display_str + "({:.3f} secs/case)"
    #                     .format(self._timer.average_time))
    #
    #     return results
    #
    # def _evaluate_images_do_mirror(self, predicts, cases=None, verbose=0, save=False, do_eval=True):
    #     # process a 3D image slice by slice
    #     logits3d = []
    #     # bg_masks3d = list()
    #     self.clear_metrics()
    #
    #     cur_case = None
    #     reshape_ori = None
    #
    #     self._timer.reset()
    #     self._timer.tic()
    #     for predict, labels in predicts:
    #         if predict is not None:
    #             new_case = str(predict["names"])
    #             cur_case = cur_case or new_case
    #             if "reshape_ori" in predict and reshape_ori is None:
    #                 reshape_ori = predict["reshape_ori"]
    #
    #             assert cur_case == new_case
    #             # Append batch to collections
    #             if predict["mirror"] == 0:
    #                 logits3d.append(predict["Prob"] / self.mirror_div)
    #             elif predict["mirror"] == 1:
    #                 logits3d[-1] += np.flip(predict["Prob"], axis=2) / self.mirror_div
    #             elif predict["mirror"] == 2:
    #                 logits3d[-1] += np.flip(predict["Prob"], axis=1) / self.mirror_div
    #             elif predict["mirror"] == 3:
    #                 logits3d[-1] += np.flip(np.flip(predict["Prob"], axis=2), axis=1) / self.mirror_div
    #         else:
    #             assert labels is not None
    #             assert isinstance(labels, tuple)
    #             labels, pads, bbox = labels
    #             volume = np.concatenate(logits3d)
    #             volume = np.argmax(volume, axis=-1)
    #             volume = {cls: volume == i + 1 for i, cls in enumerate(self.classes)}
    #             if labels is not None:
    #                 labels = np.concatenate(labels)
    #                 labels = {cls: labels == i + 1 for i, cls in enumerate(self.classes)}
    #             result = self._evaluate_case(volume, labels, cur_case, pads, bbox, save,
    #                                          False, reshape_ori, do_eval)
    #             self._timer.toc()
    #             if verbose:
    #                 log_str = ("Evaluate-{} {}".format(self._timer.calls, cur_case)
    #                            if results else "Predict-{} {}".format(self._timer.calls, cur_case))
    #                 for key, value in result.items():
    #                     log_str += " {}: {:.3f}".format(key, value)
    #                 if verbose == 1:
    #                     tf.logging.debug(log_str + " ({:.3f} s)".format(self._timer.diff))
    #                 elif verbose == 2:
    #                     tf.logging.info(log_str + " ({:.3f} s)".format(self._timer.diff))
    #
    #             logits3d.clear()
    #             labels3d.clear()
    #             logits3d.append(predict["Prob"])
    #             labels3d.append(predict["labels"])
    #
    #             if cases is not None and self._timer.calls >= cases:
    #                 break
    #
    #             # Reset
    #             cur_case = new_case
    #             pad = predict["pads"]
    #             if "bboxes" in predict:
    #                 bbox = predict["bboxes"]
    #             if "reshape_ori" in predict:
    #                 reshape_ori = predict["reshape_ori"]
    #             self._timer.tic()
    #
    #     if cases is None or (self._timer.calls < cases):
    #         # Final case
    #         volume = np.concatenate(logits3d)
    #         volume = np.argmax(volume, axis=-1)
    #         labels = np.concatenate(labels3d)
    #         volume = {cls: volume == i + 1 for i, cls in enumerate(self.classes)}
    #         labels = {cls: labels == i + 1 for i, cls in enumerate(self.classes)}
    #         result = self._evaluate_case(volume, labels, cur_case, pad, bbox, save,
    #                                      False, reshape_ori, do_eval)
    #         self._timer.toc()
    #         if verbose:
    #             log_str = ("Evaluate-{} {}".format(self._timer.calls, cur_case)
    #                        if results else "Predict-{} {}".format(self._timer.calls, cur_case))
    #             for key, value in result.items():
    #                 log_str += " {}: {:.3f}".format(key, value)
    #             if verbose == 1:
    #                 tf.logging.debug(log_str + " ({:.3f} s)".format(self._timer.diff))
    #             else:
    #                 tf.logging.info(log_str + " ({:.3f} s)".format(self._timer.diff))
    #
    #     # Compute average metrics
    #     display_str = "----Process {} cases ".format(self._timer.calls)
    #     results = {key: np.mean(values) for key, values in self._metric_values.items()}
    #     for key, value in results.items():
    #         display_str += "- {}: {:.3f} ".format(key, value)
    #     tf.logging.info(display_str + "({:.3f} secs/case)"
    #                     .format(self._timer.average_time))
    #
    #     return results

    def _evaluate_patches(self, predicts, cases=None, verbose=0, save=False):
        # process a 3D image patch by patch
        self.clear_metrics()
        self._timer.reset()

        result = None
        num_samples = None
        self._timer.tic()

        for predict in predicts:
            positions = predict["position"]
            lab = predict["labels"]     # labels will be None except the final batch of each case
            bbox = predict["bbox"]
            name = predict["name"]
            pad = predict["pad"]

            if result is None:
                result = np.zeros(arr_ops.bbox_to_shape(bbox) + (len(self.classes) + 1,),
                                  dtype=np.float32)
                num_samples = np.zeros_like(result, dtype=np.float32)

            end_id = len(positions) - pad
            for i, (z, lb_y, ub_y, lb_x, ub_x) in enumerate(positions[:end_id]):
                result[z, lb_y:ub_y, lb_x:ub_x] = predict["Prob"][i]
                num_samples[z, lb_y:ub_y, lb_x:ub_x] += 1

            if lab is not None:
                labels3d_crop = lab[arr_ops.bbox_to_slices(bbox)]
                # Finish all the patches of current case
                softmax_pred = result / num_samples
                prediction = np.argmax(softmax_pred, axis=-1)
                logits3d = {cls: prediction == i + 1 for i, cls in enumerate(self.classes)}
                labels3d = {cls: labels3d_crop == i + 1 for i, cls in enumerate(self.classes)}
                result = self._evaluate_case(logits3d, labels3d, name, 0, bbox, save,
                                             concat=False, reshape_ori=False)
                self._timer.toc()

                # Logging
                if verbose:
                    log_str = "Evaluate-{} {}".format(self._timer.calls, name)
                    for key, value in result.items():
                        log_str += " {}: {:.3f}".format(key, value)
                    if verbose == 1:
                        tf.logging.debug(log_str + " ({:.3f} s)".format(self._timer.diff))
                    elif verbose == 2:
                        tf.logging.info(log_str + " ({:.3f} s)".format(self._timer.diff))

                if cases is not None and self._timer.calls >= cases:
                    break

                # Reset
                result = None
                num_samples = None
                self._timer.tic()

        # Compute average metrics
        display_str = "----Evaluate {} cases ".format(self._timer.calls)
        results = {key: np.mean(values) for key, values in self._metric_values.items()}
        for key, value in results.items():
            display_str += "- {}: {:.3f} ".format(key, value)
        tf.logging.info(display_str + "({:.3f} secs/case)".format(self._timer.average_time))

        return results

    # def run_with_session(self, session=None, cases=None):
    #     predicts = ["labels", "names", "pads", "bboxes"] + \
    #                list(self.params["model_instances"][0].predictions)
    #     predict_gen = self.estimator.evaluate_online(session, predicts, yield_single_examples=False)
    #     tf.logging.info("Begin evaluating 3D ...")
    #     return self._evaluate(predict_gen, cases=cases, verbose=2)
    # def evaluate(self,
    #              input_fn,
    #              predict_keys=None,
    #              hooks=None,
    #              checkpoint_path=None,
    #              cases=None):
    #     if not self.params["args"].use_fewer_guide:
    #         predict_gen = self.estimator.predict(input_fn, predict_keys, hooks, checkpoint_path,
    #                                          yield_single_examples=False)
    #         tf.logging.info("Begin evaluating ...")
    #         return self._evaluate(predict_gen, cases=cases, verbose=True,
    #                               save=self.params["args"].save_predict)
    #     else:
    #         # Construct model with batch_size = 1
    #         # Disconnect input pipeline with model pipeline
    #         self.params["args"].batch_size = 1
    #         predict_gen = self.estimator.predict_with_guide(input_fn, predict_keys, hooks,
    #                                                     checkpoint_path, yield_single_examples=False)
    #         tf.logging.info("Begin evaluating ...")
    #         return self._evaluate(predict_gen, cases=cases, verbose=True,
    #                               save=self.params["args"].save_predict)

    def _predict_case(self, predicts, cases=-1, dtype="pred", resize=False, save_path=None):
        """

        Parameters
        ----------
        predicts: generator
        cases: int
        dtype: str
            prob(after softmax) or pred(after argmax)
        resize: bool
            Resize to original shape
        save_path: str or None

        Returns
        -------
        volume: with shape cshape
        segmentation: with shape cshape
        post_processed

        """
        logits3d = []
        cur_case = None
        if save_path:
            save_path = Path(save_path)
        counter = 0

        for predict, labels in predicts:
            if predict is not None:
                new_case = str(predict["names"])
                cur_case = cur_case or new_case
                assert cur_case == new_case, (cur_case, new_case)
                # Append batch to collections
                if "mirror" not in predict or predict["mirror"] == 0:
                    logits3d.append(predict["Prob"] / self.mirror_div)
                elif predict["mirror"] == 1:
                    logits3d[-1] += np.flip(predict["Prob"], axis=2) / self.mirror_div
                elif predict["mirror"] == 2:
                    logits3d[-1] += np.flip(predict["Prob"], axis=1) / self.mirror_div
                elif predict["mirror"] == 3:
                    logits3d[-1] += np.flip(np.flip(predict["Prob"], axis=2), axis=1) / self.mirror_div
            else:
                assert isinstance(labels, tuple), type(labels)
                segmentation, vol_path, pads, bbox, reshape_ori = labels
                volume = np.concatenate(logits3d)   # [d, h, w, c]
                if pads > 0:
                    volume = volume[:-pads]
                if dtype == "pred":
                    volume = np.argmax(volume, axis=-1).astype(np.uint8)    # [d, h, w]
                if resize and reshape_ori:
                    ori_shape = (volume.shape[0],) + arr_ops.bbox_to_shape(bbox)[1:]
                    if volume.ndim == 4:
                        ori_shape = ori_shape + (volume.shape[-1],)
                    scales = np.array(ori_shape) / np.array(volume.shape)
                    if np.any(scales != 1):
                        volume = ndi.zoom(volume, scales, order=0 if dtype == "pred" else 1)
                yield (cur_case, segmentation) + self.maybe_save_case(
                    cur_case, vol_path, volume, bbox, dtype, save_path)

                logits3d.clear()
                cur_case = None
                counter += 1
                if 0 < cases <= counter:
                    break

    def _postprocess(self, volume, is_label=False, ori_shape=None):
        if not isinstance(volume, dict):
            decouple_volume = {cls: volume == i + 1 for i, cls in enumerate(self.classes)}
        else:
            decouple_volume = volume
        if ori_shape is not None:
            cur_shape = decouple_volume[self.classes[0]].shape
            ori_shape = [cur_shape[0]] + list(ori_shape)
            scales = np.array(ori_shape) / np.array(cur_shape)
            for cls in self.classes:
                decouple_volume[cls] = ndi.zoom(decouple_volume[cls], scales, order=0)
        # Add tumor to liver volume
        if self.merge_tumor_to_liver and "Tumor" in decouple_volume and "Liver" in decouple_volume:
            decouple_volume["Liver"] += decouple_volume["Tumor"]

        # Find largest component --> for liver
        if self.largest and "Liver" in decouple_volume and not is_label:
            decouple_volume["Liver"] = arr_ops.get_largest_component(decouple_volume["Liver"], rank=3)
            if self.merge_tumor_to_liver and "Tumor" in decouple_volume:
                # Remove false positives outside liver region
                decouple_volume["Tumor"] *= decouple_volume["Liver"].astype(decouple_volume["Tumor"].dtype)

        return decouple_volume

    def run(self, input_fn, checkpoint_path=None, latest_filename=None, save=False):
        checkpoint_path = self.find_checkpoint_path(checkpoint_path, latest_filename)
        if not checkpoint_path:
            raise FileNotFoundError("Missing checkpoint file in {} with status_file {}".format(
                self.model_dir, latest_filename))
        model_args = self.params.get("model_args", ())
        model_kwargs = self.params.get("model_kwargs", {})

        use_context = hasattr(self.config, "use_context") and self.config.use_context
        use_spatial = hasattr(self.config, "use_spatial") and self.config.use_spatial
        # Parse context_list
        feat_length = 0
        if use_context and self.config.context_list is not None:
            if len(self.config.context_list) % 2 != 0:
                raise ValueError("context_list is not paired!")
            for i in range(len(self.config.context_list) // 2):
                feat_length += int(self.config.context_list[2 * i + 1])

        def run_pred():
            with tf.Graph().as_default():
                bs, h, w, c = (self.config.batch_size,
                               self.config.im_height if self.config.im_height > 0 else None,
                               self.config.im_width if self.config.im_width > 0 else None,
                               self.config.im_channel)
                images = tf.placeholder(tf.float32, shape=(bs, h, w, c))

                context, sp_guide = None, None
                if use_context:
                    context = tf.placeholder(tf.float32, shape=(bs, feat_length))
                if use_spatial:
                    sp_guide = tf.placeholder(tf.float32, shape=(bs, h, w, 1))

                model_inputs = {"images": images, "context": context, "sp_guide": sp_guide}
                model = self.params["model"](self.config)
                self.params["model_instances"] = [model]
                # build model
                model(model_inputs, self.config.mode, *model_args, **model_kwargs)
                saver = tf.train.Saver()
                sess = tf.Session()
                # load weights
                sess.run(tf.global_variables_initializer())
                saver.restore(sess, checkpoint_path)

                predictions = {"Prob": model.probability}
                for features, labels in input_fn(self.config.mode, self.params):
                    if features:
                        feed_dict = {images: features.pop("images")}
                        if use_context and "context" in features:
                            feed_dict[context] = features.pop("context")
                        if use_spatial and "sp_guide" in features:
                            feed_dict[sp_guide] = features.pop("sp_guide")
                        preds_eval = sess.run(predictions, feed_dict)
                        preds_eval.update(features)
                        yield preds_eval, None
                    else:
                        yield None, labels

        if self.eval_in_patches:
            tf.logging.info("Eval in patches ...")
            return self._evaluate_patches(run_pred(), cases=self.config.eval_num, verbose=2, save=save)
        else:
            resize = self.config.im_height > 0 and self.config.im_width > 0
            self._run_actual(self._predict_case, run_pred, save, resize=resize)

    def _predict_case_g(self, predicts, cases=-1, dtype="pred", save_path=None):
        logits3d = defaultdict(list)

        cur_case = None
        if save_path:
            save_path = Path(save_path)
        counter = 0

        for predict, labels in predicts:
            if predict is not None:
                new_case = str(predict["names"])
                cur_case = cur_case or new_case
                di = predict["direction"]
                assert cur_case == new_case, (cur_case, new_case)
                # Append batch to collections
                if predict["mirror"] == 0:
                    logits3d[di].append(predict["Prob"] / self.mirror_div)
                elif predict["mirror"] == 1:
                    logits3d[di][-1] += np.flip(predict["Prob"], axis=2) / self.mirror_div
                elif predict["mirror"] == 2:
                    logits3d[di][-1] += np.flip(predict["Prob"], axis=1) / self.mirror_div
                elif predict["mirror"] == 3:
                    logits3d[di][-1] += np.flip(np.flip(predict["Prob"], axis=2), axis=1) / self.mirror_div
            else:
                assert isinstance(labels, tuple), type(labels)
                segmentation, vol_path, pads, bbox = labels
                volume = np.concatenate(logits3d["Forward"], axis=0)    # [d, h, w, c]
                if "Backward" in logits3d:
                    volume_rev = np.concatenate(logits3d["Backward"], axis=0)
                    volume = np.maximum(volume, np.flip(volume_rev, axis=0))
                if pads > 0:
                    volume = volume[:-pads]
                if dtype == "pred":
                    volume = np.argmax(volume, axis=-1).astype(np.uint8)    # [d, h, w]
                if True:
                    ori_shape = (volume.shape[0],) + arr_ops.bbox_to_shape(bbox)[1:]
                    if volume.ndim == 4:
                        ori_shape += (volume.shape[-1],)
                    scales = np.array(ori_shape) / np.array(volume.shape)
                    if np.any(scales != 1):
                        volume = ndi.zoom(volume, scales, order=0 if dtype == "pred" else 1)
                yield (cur_case, segmentation) + self.maybe_save_case(
                    cur_case, vol_path, volume, bbox, dtype, save_path)

                logits3d.clear()
                cur_case = None
                counter += 1
                if 0 < cases <= counter:
                    break

    def run_g(self, input_fn, checkpoint_path=None, latest_filename=None, save=False):
        checkpoint_path = self.find_checkpoint_path(checkpoint_path, latest_filename)
        if not checkpoint_path:
            raise FileNotFoundError("Missing checkpoint file in {} with status_file {}".format(
                self.model_dir, latest_filename))
        model_args = self.params.get("model_args", ())
        model_kwargs = self.params.get("model_kwargs", {})

        # Parse context_list
        feat_length = 0
        if self.config.use_context and self.config.context_list is not None:
            if len(self.config.context_list) % 2 != 0:
                raise ValueError("context_list is not paired!")
            for i in range(len(self.config.context_list) // 2):
                feat_length += int(self.config.context_list[2 * i + 1])

        def run_pred():
            with tf.Graph().as_default():
                # Force batch size to be 1 if enable use_spatial_guide
                bs, h, w, c = (1 if self.config.use_spatial else self.config.batch_size,
                               self.config.im_height if self.config.im_height > 0 else None,
                               self.config.im_width if self.config.im_width > 0 else None,
                               self.config.im_channel)
                images = tf.placeholder(tf.float32, shape=(bs, h, w, c))

                context, sp_guide = None, None
                if self.config.use_context:
                    context = tf.placeholder(tf.float32, shape=(bs, feat_length))
                if self.config.use_spatial:
                    sp_guide = tf.placeholder(tf.float32, shape=(bs, h, w, 1))

                model_inputs = {"images": images, "context": context, "sp_guide": sp_guide}
                model = self.params["model"](self.config)
                self.params["model_instances"] = [model]
                # build model
                model(model_inputs, self.config.mode, *model_args, **model_kwargs)
                saver = tf.train.Saver()
                sess = tf.Session()
                # load weights
                sess.run(tf.global_variables_initializer())
                saver.restore(sess, checkpoint_path)

                predictions = {"Prob": model.probability, "TumorPred": model.predictions["TumorPred"]}

                if self.config.use_spatial:
                    eil = input_fn(self.config.mode, self.params)  # EvalImage3DLoader
                    # TODO(zjw) remove these lines
                    from DataLoader.Liver import input_pipeline_g
                    assert isinstance(eil, input_pipeline_g.EvalImage3DLoader)
                    while eil.prepare_next_case():
                        for slice_iter in eil.case_iter:
                            # slice_preds = []
                            slice_probs = []
                            for features in slice_iter:
                                feed_dict = {images: features.pop("images"), sp_guide: features.pop("sp_guide")}
                                if self.config.use_context:
                                    feed_dict[context] = features.pop("context")
                                preds_eval = sess.run(predictions, feed_dict=feed_dict)
                                preds_eval.update(features)
                                if features["mirror"] == 0:
                                    # slice_preds.append(preds_eval.pop("TumorPred"))
                                    slice_probs.append(preds_eval["Prob"])
                                elif features["mirror"] == 1:
                                    # slice_preds.append(np.flip(preds_eval.pop("TumorPred"), axis=2))
                                    slice_probs.append(np.flip(preds_eval["Prob"], axis=2))
                                elif features["mirror"] == 2:
                                    # slice_preds.append(np.flip(preds_eval.pop("TumorPred"), axis=1))
                                    slice_probs.append(np.flip(preds_eval["Prob"], axis=1))
                                elif features["mirror"] == 3:
                                    # slice_preds.append(np.flip(np.flip(preds_eval.pop("TumorPred"), axis=2), axis=1))
                                    slice_probs.append(np.flip(np.flip(preds_eval["Prob"], axis=2), axis=1))
                                yield preds_eval, None
                            # ori_dtype = slice_preds[0].dtype
                            # eil.last_pred = np.ceil(np.mean(slice_preds, axis=0)).astype(ori_dtype)
                            eil.last_pred = (np.argmax(np.mean(slice_probs, axis=0), axis=-1) == 2)\
                                .astype(np.uint8)[..., None]
                        yield None, eil.labels
                else:
                    for features, labels in input_fn(self.config.mode, self.params):
                        if features:
                            preds_eval = sess.run(predictions, {images: features.pop("images"),
                                                                context: features.pop("context")})
                            preds_eval.update(features)
                            yield preds_eval, None
                        else:
                            yield None, labels
        self._run_actual(self._predict_case_g, run_pred, save)

    def _run_actual(self, predict_fn, run_fn, save, **run_kwargs):
        if self.config.mode == ModeKeys.EVAL:
            tf.logging.info("Eval in images ...")
            do_eval = True
        else:  # self.config.mode == ModeKeys.PREDICT
            tf.logging.info("Predict in images ...")
            do_eval = False
        if save:
            if self.config.save_path:
                save_path = Path(self.model_dir) / self.config.save_path
            else:
                save_path = Path(self.model_dir) / "prediction"
            save_path.mkdir(parents=True, exist_ok=True)
        else:
            save_path = None

        self._timer.reset()
        accumulator = defaultdict(int)
        if not self.config.use_global_dice:
            self.clear_metrics()
            self._timer.tic()
            for cur_case, labels, volume, post_processed in predict_fn(run_fn(),
                                                                       cases=self.config.eval_num,
                                                                       dtype=self.config.pred_type,
                                                                       save_path=save_path,
                                                                       **run_kwargs):
                self._timer.toc()
                results = {}
                if do_eval:
                    if not post_processed:
                        volume = self._postprocess(volume)
                    labels = self._postprocess(labels, is_label=True)
                    # For global dice
                    for i, cls in enumerate(self.classes):
                        conf = metric_ops.ConfusionMatrix(volume[cls].astype(int), labels[cls].astype(int))
                        conf.compute()
                        accumulator[cls + "_fn"] += conf.fn
                        accumulator[cls + "_fp"] += conf.fp
                        accumulator[cls + "_tp"] += conf.tp
                    # Calculate 3D metrics
                    for c, cls in enumerate(self.classes):
                        pairs = metric_ops.metric_3d(volume[cls], labels[cls], required=self.metrics_str)
                        for met, value in pairs.items():
                            results["{}/{}".format(cls, met)] = value
                    self.append_metrics(results)
                log_str = ("Evaluate-{} {}".format(self._timer.calls, cur_case)
                           if results else "Predict-{} {}".format(self._timer.calls, cur_case))
                for key, value in results.items():
                    log_str += " {}: {:.3f}".format(key, value)
                else:
                    tf.logging.info(log_str + " ({:.3f} s)".format(self._timer.diff))
                self._timer.tic()
            results = {key: np.mean(values) for key, values in self._metric_values.items()}
            if accumulator:
                results.update({"G" + cls + "Dice": 2 * accumulator[cls + "_tp"] / (
                        2 * accumulator[cls + "_tp"] + accumulator[cls + "_fn"] + accumulator[cls + "_fp"])
                                for cls in self.classes})
        else:
            accumulator = defaultdict(int)
            self._timer.tic()
            for cur_case, labels, volume, post_processed in predict_fn(run_fn(),
                                                                       cases=self.config.eval_num,
                                                                       dtype=self.config.pred_type,
                                                                       save_path=save_path,
                                                                       **run_kwargs):
                self._timer.toc()
                results = {}
                if not post_processed:
                    volume = self._postprocess(volume)
                labels = self._postprocess(labels, is_label=True)

                for i, cls in enumerate(self.classes):
                    conf = metric_ops.ConfusionMatrix(volume[cls].astype(int), labels[cls].astype(int))
                    conf.compute()
                    accumulator[cls + "_fn"] += conf.fn
                    accumulator[cls + "_fp"] += conf.fp
                    accumulator[cls + "_tp"] += conf.tp

                log_str = ("Evaluate-{} {}".format(self._timer.calls, cur_case)
                           if results else "Predict-{} {}".format(self._timer.calls, cur_case))
                tf.logging.info(log_str + " ({:.3f} s)".format(self._timer.diff))
                self._timer.tic()
            results = {cls + "Dice": 2 * accumulator[cls + "_tp"] / (
                    2 * accumulator[cls + "_tp"] + accumulator[cls + "_fn"] + accumulator[cls + "_fp"])
                       for cls in self.classes}

        # Compute average metrics
        display_str = "----Process {} cases ".format(self._timer.calls)
        for key, value in results.items():
            display_str += "- {}: {:.3f} ".format(key, value)
        tf.logging.info(display_str + "({:.3f} secs/case)".format(self._timer.average_time))

    def maybe_save_case(self, cur_case, vol_path, volume, bbox, dtype, save_path):
        if save_path:
            case_header = nib.load(str(vol_path)).header
            pad_with = tuple(zip(bbox[2::-1],
                                 np.array(case_header.get_data_shape()[::-1]) - bbox[:2:-1] - 1))
            if dtype == "pred":
                volume = self._postprocess(volume)
                if "Liver" in volume and "Tumor" in volume:
                    img_array = volume["Liver"] + volume["Tumor"]
                elif "Liver" in volume:
                    img_array = volume["Liver"]
                elif "Tumor" in volume:
                    img_array = volume["Tumor"]
                else:
                    raise ValueError("Not supported save object!")
                img_array = np.pad(img_array, pad_with, mode="constant", constant_values=0)
                save_file = save_path / "predict-{}.nii.gz".format(cur_case)
                nii_kits.write_nii(img_array, case_header, save_file)
            else:
                # Save 4D Tensor to npz
                pad_with = pad_with + ((0, 0),)
                img_array = np.pad(volume, pad_with, mode="constant", constant_values=0)
                save_file = save_path / (cur_case + ".npz")
                np.savez_compressed(str(save_file), img_array)
            tf.logging.info("    ==> Save to {}"
                            .format(str(save_file.relative_to(save_path.parent.parent.parent))))
            return volume, dtype == "pred"  # post_processed
        else:
            return volume, False

    # @staticmethod
    # def _check_shapes_equal(volume_dict):
    #     # Don't use! Just for debug
    #     ref = {}
    #     mismatch = {}
    #     shape = None
    #     for key, value in volume_dict.items():
    #         if shape is None:
    #             shape = value.shape
    #             ref[key] = shape
    #         elif value.shape != shape:
    #             mismatch[key] = value.shape
    #     if len(mismatch) > 0:
    #         log_str = "Shape mismatch: Ref({} -> {}), Wrong(".format(*list(ref.items())[0])
    #         for key, value in mismatch.items():
    #             log_str += "{} -> {}  ".format(key, value)
    #         raise ValueError(log_str)

    def _evaluate_case(self, logits3d, labels3d, cur_case, pad, bbox=None, save=False,
                       concat=True, reshape_ori=True, do_eval=True):
        # Process a complete volume
        if concat:
            logits3d = {cls: np.concatenate(values) for cls, values in logits3d.items()}
            labels3d = {cls: np.concatenate(values) for cls, values in labels3d.items()}

        if pad != 0:
            logits3d = {cls: value[:-pad] if not cls.endswith("_rev") else value[pad:]
                        for cls, value in logits3d.items()}
            labels3d = {cls: value[:-pad] for cls, value in labels3d.items()}

        if bbox is not None and reshape_ori:
            # Resize logits3d to the shape of labels3d
            ori_shape = list(arr_ops.bbox_to_shape(bbox))
            cur_shape = logits3d[self.classes[0]].shape
            ori_shape[0] = cur_shape[0]
            scales = np.array(ori_shape) / np.array(cur_shape)
            for c, cls in enumerate(self.classes):
                logits3d[cls] = ndi.zoom(logits3d[cls], scales, order=0)

        # Add tumor to liver volume
        if self.merge_tumor_to_liver and "Tumor" in logits3d and "Liver" in logits3d:
            logits3d["Liver"] += logits3d["Tumor"]
            labels3d["Liver"] += labels3d["Tumor"]

        # Find largest component --> for liver
        if self.largest and "Liver" in logits3d:
            logits3d["Liver"] = arr_ops.get_largest_component(logits3d["Liver"], rank=3)
            if self.merge_tumor_to_liver and "Tumor" in logits3d:
                # Remove false positives outside liver region
                logits3d["Tumor"] *= logits3d["Liver"].astype(logits3d["Tumor"].dtype)

        # Remove false positives with spatial guide
        if self.use_sg_reduce_fp:
            logits3d["Tumor"] = arr_ops.reduce_fp_with_guide(np.squeeze(labels3d["Tumor"], axis=-1),
                                                             logits3d["Tumor"],
                                                             guide="middle")

        cur_pairs = {}
        if do_eval:
            # Calculate 3D metrics
            for c, cls in enumerate(self.classes):
                pairs = metric_ops.metric_3d(logits3d[cls], labels3d[cls], required=self.metrics_str)
                for met, value in pairs.items():
                    cur_pairs["{}/{}".format(cls, met)] = value
            self.append_metrics(cur_pairs)

        if save:
            seg_path = Path(__file__).parent.parent / ("data/LiTS/Training_Batch/segmentation-{}.nii"
                                                       .format(cur_case))
            case_name = seg_path.name.replace("segmentation", "prediction") + ".gz"
            save_path = Path(self.model_dir) / "prediction"
            if not save_path.exists():
                save_path.mkdir(exist_ok=True)
            save_path = save_path / case_name

            if "Liver" in logits3d and "Tumor" in logits3d:
                img_array = logits3d["Liver"] + logits3d["Tumor"]
            elif "Liver" in logits3d:
                img_array = logits3d["Liver"]
            elif "Tumor" in logits3d:
                img_array = logits3d["Tumor"]
            else:
                raise ValueError("Not supported save object!")

            case_header = nib.load(str(seg_path)).header
            pad_with = tuple(zip(bbox[2::-1], np.array(case_header.get_data_shape()[::-1]) - bbox[:2:-1] - 1))
            img_array = np.pad(img_array, pad_with, mode="constant", constant_values=0)

            nii_kits.write_nii(img_array, case_header, save_path,
                               special=True if 28 <= int(cur_case) < 52 else False)
            tf.logging.info("    ==> Save to {}"
                            .format(str(save_path.relative_to(save_path.parent.parent.parent))))
        return cur_pairs

    def compare(self, *args_, **kwargs):
        return _compare(*args_, **kwargs)


# class EvaluateSlice(EvaluateBase):
#     """ Evaluate Estimator model by slice """
#
#     def __init__(self, estimator):
#         """
#         Parameters
#         ----------
#         estimator: CustomEstimator
#             CustomEstimator instance
#         """
#         self.estimator = estimator
#         self.params = estimator.params
#
#     @property
#     def classes(self):
#         return self.params["model_instances"][0].classes[1:]  # Remove background
#
#     def run_with_session(self, session=None, cases=None):
#         predicts = list(self.params["model_instances"][0].metrics_dict.keys())
#         tf.logging.info("Begin evaluating 2D ...")
#         predict_gen = self.estimator.predict_with_session(session, predicts, yield_single_examples=False)
#
#         self.clear_metrics()
#         for predict in predict_gen:
#             self.append_metrics(predict)
#
#         results = {key: np.mean(values) for key, values in self._metric_values.items()}
#         display_str = ""
#         for key, value in results.items():
#             display_str += "{}: {:.3f} ".format(key, value)
#         tf.logging.info(display_str)
#
#         return results
#
#     def run(self, input_fn, predict_keys=None, hooks=None, checkpoint_path=None, cases=None):
#         predicts = ["Names", "Indices"]
#         tf.logging.info("Begin evaluating 2D ...")
#         if not self.params["args"].use_fewer_guide:
#             predict_gen = self.estimator.predict(input_fn, predicts, hooks, checkpoint_path,
#                                                  yield_single_examples=True)
#         else:
#             # Construct model with batch_size = 1
#             # Disconnect input pipeline with model pipeline
#             self.params["args"].batch_size = 1
#             predict_gen = self.estimator.predict_with_guide(input_fn, predicts, hooks,
#                                                             checkpoint_path, yield_single_examples=True)
#
#         self.clear_metrics()
#         for i, pred_ in enumerate(predict_gen):
#             print("\rEval {} examples ...".format(i + 1), end="")
#             self.append_metrics(pred_)
#         print()
#         self.save_metrics("metrics_2d.txt", self.estimator.model_dir)
#
#         results = {key: float(np.mean(values)) for key, values in self._metric_values.items()
#                    if key not in ["Names", "Indices"]}
#         display_str = ""
#         for key, value in results.items():
#             display_str += "{}: {:.3f} ".format(key, value)
#         tf.logging.info(display_str)
#
#         return results
#
#     def compare(self, *args_, **kwargs):
#         return _compare(*args_, **kwargs)


def _compare(cur_result,
             ori_result,
             primary_metric=None,
             secondary_metric=None):
    if not isinstance(cur_result, dict):
        raise TypeError("`cur_result` should be dict, but got {}".format(type(cur_result)))
    if not isinstance(ori_result, dict):
        raise TypeError("`ori_result` should be dict, but got {}".format(type(ori_result)))
    if set(cur_result) != set(ori_result):
        raise ValueError("Dicts with different keys can not be compared. "
                         "cur_result({}) vs ori_result({})"
                         .format(list(cur_result.keys()), list(ori_result.keys())))
    if primary_metric and primary_metric not in cur_result:
        raise KeyError("`primary_metric` not in valid result key: {}".format(primary_metric))
    if secondary_metric and secondary_metric not in cur_result:
        raise KeyError("`secondary_metric` not in valid result key: {}".format(secondary_metric))
    if primary_metric == secondary_metric:
        raise ValueError("`primary_metric` can not be equal to `secondary_metric`")

    keys = list(cur_result.keys())
    if primary_metric:
        keys.remove(primary_metric)
        keys.insert(0, primary_metric)
        if secondary_metric:
            keys.remove(secondary_metric)
            keys.insert(1, secondary_metric)

    for key in keys:
        if cur_result[key] > ori_result[key]:
            return True
        elif cur_result[key] == ori_result[key]:
            continue
        else:
            return False
    return False


# class TumorManager(object):
#     def __init__(self, tumor_info, min_std):
#         self._tumor_info = tumor_info
#         self._name = None
#         self._bbox = None
#         self._id = None
#         self.direction = 1  # 1 for "forward", -1 for "backward"
#         self.disc = ndi.generate_binary_structure(2, connectivity=1)
#         self.guides = None
#         self.pred = None
#         self.backup = defaultdict(list)
#         self.debug = False
#         self.min_std = min_std
#
#     @property
#     def info(self):
#         return self._tumor_info
#
#     @property
#     def name(self):
#         return self._name
#
#     @name.setter
#     def name(self, new_name):
#         self._name = new_name
#         self._name_id = int(Path(new_name).name.split(".")[0].split("-")[1])
#         self.total_tumors = self._tumor_info[self._tumor_info["PID"] ==
#                                              "segmentation-{}.nii".format(self._name_id)]
#         self.total_tumors = self.total_tumors.iloc[:, range(2, 8)].values
#         self.total_tumors[:, [3, 4, 5]] -= 1    # [) --> []
#         self.total_tumors_yx = self.total_tumors[:, [1, 0, 4, 3]]  # [y1, x1, y2, x2]
#         self.total_tumors_z = self.total_tumors[:, [2, 5]]          # range: [z1, z2]
#         del self.total_tumors
#
#     @property
#     def bbox(self):
#         return self._bbox
#
#     def set_bbox(self, new_bbox, shape):
#         """
#         new_bbox: make sure (x1, y1, z1, x2, y2, z2)
#         """
#         self._bbox = np.asarray(new_bbox)   # []
#         self.shape = np.asarray(shape)
#         scale = self.shape / np.array([self._bbox[4] - self._bbox[1] + 1,
#                                        self._bbox[3] - self._bbox[0] + 1])
#         self.total_tumors_yx = (self.total_tumors_yx - self._bbox[[1, 0, 1, 0]]) * np.tile(scale, [2])
#         self.total_tumors_z -= self._bbox[[2, 2]]
#
#     @property
#     def id(self):
#         """ slice id in CT """
#         return self._id
#
#     def _set_id(self, new_id):
#         if self._name is None or self._bbox is None:
#             raise ValueError("Please set name and bbox first")
#         self._id = new_id
#
#     def clear_backup(self):
#         for key in self.backup:
#             self.backup[key].clear()
#
#     def append_backup(self, center, stddev, zi):
#         self.backup["centers"].append(center)
#         self.backup["stddevs"].append(stddev)
#         self.backup["zi"].append(zi)
#
#     def reset(self, direction=1):
#         self._name = None
#         self._bbox = None
#         self._id = None
#         self.direction = direction
#         self.guides = None
#         self.pred = None
#         self.clear_backup()
#
#     def print(self, *args_, **kwargs):
#         if self.debug:
#             print(*args_, **kwargs)
#
#     def set_guide_info(self, guide, new_id):
#         """
#         Decide whether to finish a tumor or not.
#         Of course making this decision with only guide is enough.
#
#         Parameters
#         ----------
#         guide: Tensor
#             with shape [n, 4]
#         new_id: int
#             Index number of current slice in a CT volume.
#         """
#         self._set_id(new_id)
#
#         select = np.where(guide[:, 0] >= 0)[0]
#         self.centers_yx = np.round(guide[select, 1::-1] * self.shape).astype(np.int32)
#         self.stddevs_yx = np.round(guide[select, :1:-1] * self.shape).astype(np.int32)
#         # print(guide)
#         # Indices for self.total_tumor_z
#         self.zi = np.array([self.determine_z_min_max(i, ctr, std)
#                             for i, (ctr, std) in enumerate(zip(self.centers_yx, self.stddevs_yx))],
#                            dtype=np.int32)
#         self.print("{} New Guide: {}".format(new_id + self._bbox[2], self.zi.shape[0]), end="")
#         if self.backup["zi"]:
#             # TODO(ZJW): Maybe we should remove the same objects from centers_ys and centers_backup.
#             #            For example, two or more adjacent slices are provided guides.
#             self.centers_yx = np.concatenate(
#                 (self.centers_yx, np.asarray(self.backup["centers"], np.int32)), axis=0)
#             self.stddevs_yx = np.concatenate(
#                 (self.stddevs_yx, np.asarray(self.backup["stddevs"], np.int32)), axis=0)
#             self.zi = np.concatenate(
#                 (self.zi, np.asarray(self.backup["zi"], np.int32)), axis=0)
#         self.print("  Last Guide: {}".format(len(self.backup["zi"])))
#
#     def get_guide_image(self, guide, new_id):
#         self.set_guide_info(guide, new_id)
#
#         if len(self.centers_yx) > 0:
#             self.guides = arr_ops.create_gaussian_distribution(self.shape,
#                                                                self.centers_yx[0, ::-1],
#                                                                self.stddevs_yx[0, ::-1])
#             for i in range(1, len(self.centers_yx)):
#                 self.guides = np.maximum(self.guides, arr_ops.create_gaussian_distribution(
#                     self.shape, self.centers_yx[i, ::-1], self.stddevs_yx[i, ::-1]))
#         else:
#             self.guides = np.zeros(self.shape, dtype=np.float32)
#
#         if self.pred is None:
#             return self.guides[None, ..., None]
#         else:
#             self.guides = np.maximum(
#                 self.guides, arr_ops.get_gd_image_multi_objs(
#                     self.pred, center_perturb=0., stddev_perturb=0.))
#             return self.guides[None, ..., None]
#
#     def check_pred(self, predict, filter_thresh=0.15):
#         """
#         Remove those predicted tumors who are out of range.
#         Apply supervisor to predicted tumors.
#
#         Make sure `pred` is binary
#
#         self.pred which is created in this function will be used for generating next guide.
#         So don't use `self.pred` as real prediction, because we will also remove those ended
#         in current slice from self.pred.
#
#         TODO(ZJW): adjust filter_thresh 0.35 ?
#         """
#         if self.guides is None:
#             raise ValueError("previous_guide is None")
#         if np.sum(predict) == 0:
#             return predict
#
#         self.clear_backup()
#
#         labeled_objs, n_objs = ndi.label(predict, self.disc)
#         slicers = ndi.find_objects(labeled_objs)
#         # Decide whether reaching the end of the tumor or not
#         for i, slicer in zip(range(n_objs), slicers):
#             res_obj = labeled_objs == i + 1
#             res_obj_slicer = res_obj[slicer]
#             # 1. Filter wrong tumors(no corresponding guide)
#             mask_guide_by_res = res_obj_slicer * self.guides[slicer]
#             # print(np.max(mask_guide_by_res))
#             if np.max(mask_guide_by_res) < filter_thresh:
#                 self.print("Remove")
#                 predict[slicer] -= res_obj_slicer   # Faster than labeled_objs[res_obj] = 0
#                 continue
#             # 2. Match res_obj to guide
#             res_peak_pos = list(np.unravel_index(mask_guide_by_res.argmax(), mask_guide_by_res.shape))
#             res_peak_pos[0] += slicer[0].start
#             res_peak_pos[1] += slicer[1].start
#             #   2.1. Check whether res_peak is just a guide center
#             found = -1
#             for j, center in enumerate(self.centers_yx):
#                 if res_peak_pos[0] == center[0] and res_peak_pos[1] == center[1]:
#                     found = j   # res_peak is just a center
#                     break
#             #   2.2. From the nearest guide center, check that whether it is the corresponding guide.
#             #        Rule: Image(guide) values along the line from res_obj's peak to its corresponding
#             #        guide center must be monotonously increasing.
#             if found < 0:   # gradient ascent from res_peak to center
#                 # compute distances between res_obj_peak and every guide center
#                 distances = np.sum((self.centers_yx - res_peak_pos) ** 2, axis=1)
#                 order = np.argsort(distances)
#                 for j in order:
#                     ctr = self.centers_yx[j]
#                     if self.ascent_line(self.guides, res_peak_pos[1], res_peak_pos[0], ctr[1], ctr[0]):
#                         # Found
#                         found = j
#                         break
#             if found < 0:
#                 raise ValueError("Can not find corresponding guide!")
#             # 3. Check z range and stop finished tumors(remove from next guide image)
#             if (self.direction == 1 and self._id >= self.total_tumors_z[self.zi[found]][1]) or \
#                     (self.direction == -1 and self._id <= self.total_tumors_z[self.zi[found]][0]):
#                 # if self.direction == -1:
#                 #     print("End {} vs {}, {}".format(self._id, self.total_tumors_z[self.zi[found]][0],
#                 #                                     self._bbox[2]))
#                 # if self.direction == 1:
#                 #     print("End {} vs {}, {}".format(self._id, self.total_tumors_z[self.zi[found]][1],
#                 #                                     self._bbox[2]))
#                 predict[slicer] -= res_obj_slicer
#                 continue
#             # 4. Compute moments. Save moments of tumors for next slice
#             ctr, std = arr_ops.compute_robust_moments(res_obj_slicer, indexing="ij", min_std=self.min_std)
#             ctr[0] += slicer[0].start
#             ctr[1] += slicer[1].start
#             self.append_backup(ctr, std, self.zi[found])
#             # print(ctr, std, self.zi[found])
#
#         self.pred = predict
#
#     @staticmethod
#     def ascent_line(img, x0, y0, x1, y1):
#         # Find points along this line
#         xs, ys, forward = arr_ops.xiaolinwu_line(x0, y0, x1, y1)
#         ascent = True
#         pre = img[ys[0], xs[0]] if forward else img[ys[-1], xs[-1]]
#         xs, ys = (xs, ys) if forward else (reversed(xs[:-1]), reversed(ys[:-1]))
#         for x, y in zip(xs, ys):
#             cur = img[y, x]
#             if cur >= pre:
#                 pre = cur
#                 continue
#             else:
#                 ascent = False
#                 break
#         return ascent
#
#     def determine_z_min_max(self, idx, center, stddev):
#         _ = stddev  # Unused
#         diff = self.total_tumors_yx - np.tile(center, [2])
#         sign = np.all(diff[:, 0:2] * diff[:, 2:4] <= 0, axis=1)
#         select = np.where(sign)[0]
#         if len(select) == 0:
#             nn_dist = np.mean(np.abs(diff), axis=1)
#             nn_idx = np.argmin(nn_dist)
#             if nn_dist[nn_idx] < 2.:
#                 return nn_idx
#             else:
#                 print("#" * 50)
#                 print(nn_idx, nn_dist[nn_idx])
#                 print(self._id, idx, center)
#                 print(self._bbox)
#                 print(self.total_tumors_yx)
#                 print("#" * 50)
#                 raise ValueError
#         elif len(select) == 1:
#             return select[0]
#         else:
#             tumor_centers = (self.total_tumors_yx[select, 0:2] + self.total_tumors_yx[select, 2:4]) / 2
#             distance = np.sum((tumor_centers - center) ** 2, axis=1)
#             new_sel = np.argmin(distance)
#             return select[new_sel]


# if __name__ == "__main__":
#     import pandas as pd
#     import input_pipeline_osmn
#     import matplotlib.pyplot as plt
#     from utils import nii_kits
#
#     tumor_path = Path(__file__).parent / "data/LiTS/tumor_summary.csv"
#     tumors_info = pd.read_csv(str(tumor_path))
#
#     class Foo(object):
#         pass
#
#     args = Foo()
#     args.input_group = 3
#     args.eval_skip_num = 20
#     args.batch_size = 1
#     args.num_gpus = 1
#     args.guide = "middle"
#     args.resize_for_batch = True
#     args.im_height = 256
#     args.im_width = 256
#     args.hist_scale = 1.0
#     args.w_width = 450
#     args.w_level = 50
#
#     n = 0
#     dataset = input_pipeline_osmn.get_3d_multi_records_dataset_for_eval(
#         [r"D:\documents\MLearning\MultiOrganDetection\core\MedicalImageSegmentation"
#          r"\data\LiTS\records\trainval-bbox-3D-3-of-5.tfrecord"],
#         [r"D:\documents\MLearning\MultiOrganDetection\core\MedicalImageSegmentation"
#          r"\data\LiTS\records\hist-100--200_250-3D-3-of-5.tfrecord"],
#         mode="eval",
#         args=args
#     ).skip(n).make_one_shot_iterator().get_next()
#
#     def run(mgr):
#         _, temp = nii_kits.nii_reader(Path(__file__).parent / "model_dir/016_osmn_in_noise"
#                                                               "/prediction/prediction-113.nii.gz")
#         temp = arr_ops.merge_labels(temp, [0, 2])
#         temp = temp[arr_ops.bbox_to_slices(mgr.bbox)]
#         temp = ndi.zoom(temp, [1, mgr.shape[0] / temp.shape[1], mgr.shape[1] / temp.shape[2]],
#                         order=0)[n:]
#         for x in np.concatenate((temp, np.flip(temp, axis=0)), axis=0):
#             yield x
#
#     sess = tf.Session()
#     features_val, labels_val = sess.run(dataset)
#     # print(features_val["names"], np.clip(labels_val - 1, 0, 1).sum(), features_val["sp_guide"])
#     # features_val, labels_val = sess.run(dataset)
#     # print(features_val["names"], np.clip(labels_val - 1, 0, 1).sum(), features_val["sp_guide"])
#
#     t_mgr = TumorManager(tumors_info)
#     t_mgr.name = features_val["names"][0].decode("utf-8")
#     z0 = features_val["bboxes"][0][2]
#     t_mgr.set_bbox(features_val["bboxes"][0], shape=features_val["images"].shape[1:-1])
#     features_val["sp_guide"] = t_mgr.get_guide_image(features_val["sp_guide"][0], new_id=n)
#     run_gen = run(t_mgr)
#
#     fig, ax = plt.subplots(1, 2)
#     init = next(run_gen)
#     t_mgr.check_pred(init)
#     init = init.copy()
#     init[0, 0] = 1
#     init2 = features_val["sp_guide"][0, ..., 0]
#     init2[0, 0] = 1
#     spg_handle = ax[0].imshow(init2, cmap="gray")
#     img_handle = ax[1].imshow(init, cmap="gray")
#     text = plt.title("{}".format(n + z0))
#     n += 1
#
#     cur_name = features_val["names"][0]
#
#     def key_press_event(event):
#         global n, features_val, labels_val, pred, cur_name
#         if event.key == "down":
#             features_val, labels_val = sess.run(dataset)
#             new_name = features_val["names"][0]
#             if new_name != cur_name:
#                 print(new_name.decode("utf-8"))
#                 cur_name = new_name
#                 n = n - 1
#                 t_mgr.reset(-1)
#                 t_mgr.name = cur_name.decode("utf-8")
#                 t_mgr.set_bbox(features_val["bboxes"][0], features_val["images"].shape[1:-1])
#             features_val["sp_guide"] = t_mgr.get_guide_image(features_val["sp_guide"][0], new_id=n)
#             pred = next(run_gen)
#             t_mgr.check_pred(pred)
#             spg_handle.set_data(features_val["sp_guide"][0, ..., 0])
#             img_handle.set_data(pred)
#             text.set_text("{}".format(n + z0))
#             if t_mgr.direction == 1:
#                 n += 1
#             else:
#                 n -= 1
#             fig.canvas.draw()
#
#     fig.canvas.mpl_connect("key_press_event", key_press_event)
#     plt.show()
