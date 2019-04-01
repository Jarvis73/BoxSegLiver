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

import numpy as np
import tensorflow as tf
import scipy.ndimage as ndi
from collections import defaultdict
from pathlib import Path

import loss_metrics as metric_ops
import utils.array_kits as arr_ops
import data_kits.build_data as data_ops
from custom_estimator import CustomEstimator
from custom_evaluator_base import EvaluateBase
from utils.timer import Timer


def add_arguments(parser):
    group = parser.add_argument_group(title="Evaluation Arguments")
    group.add_argument("--save_best",
                       action="store_false",
                       required=False, help="Save checkpoint with best results")
    group.add_argument("--eval_steps",
                       type=int,
                       default=5000,
                       required=False, help="Evaluate pre several steps (default: %(default)d)")
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
    group.add_argument("--use_fewer_guide",
                       action="store_true",
                       required=False, help="Use fewer guide for evaluation")
    group.add_argument("--guide",
                       type=str,
                       default="first",
                       choices=["first", "middle"],
                       required=False, help="Generate guide from which slice")
    group.add_argument("--eval_num",
                       type=int,
                       required=False, help="Number of cases for evaluation")
    group.add_argument("--eval_skip_num",
                       type=int,
                       default=0,
                       required=False, help="Skip some cases for evaluating determined case")
    group.add_argument("--eval_3d",
                       action="store_true",
                       required=False, help="Evaluate when training in 2D slices or 3D volume."
                                            "Default in 2D slices")


def get_eval_params(evaluator="Volume",
                    eval_steps=2500,
                    largest=True,
                    merge_tumor_to_liver=True,
                    primary_metric=None,
                    secondary_metric=None):
    if evaluator not in ["Volume", "Slice"]:
        raise ValueError("Unsupported evaluator: {}. Must be [Volume, Slice]".format(evaluator))
    return {
        "evaluator": eval("Evaluate" + evaluator),
        "eval_steps": eval_steps,
        "eval_kwargs": {"merge_tumor_to_liver": merge_tumor_to_liver, "largest": largest},
        "primary_metric": primary_metric,
        "secondary_metric": secondary_metric
    }


class EvaluateVolume(EvaluateBase):
    """ Evaluate Estimator model by volume

    This class is for liver and tumor segmentation.
    Inherit this class and rewrite self._evaluate_case() for custom dataset.
    """
    def __init__(self, model, **kwargs):
        """
        Parameters
        ----------
        model: CustomEstimator
            CustomEstimator instance
        kwargs: dict
            * merge_tumor_to_liver: bool, if `Tumor` and `Liver` in predictions (default True)
            * largest: bool, get largest component for liver, if `Liver` in predictions (default True)
        """
        if not isinstance(model, CustomEstimator):
            raise TypeError("model need a custom_estimator.CustomEstimator instance")
        super(EvaluateVolume, self).__init__(model, **kwargs)

        self.model = model
        # self._metric_values = {"{}/{}".format(cls, met): []
        #                        for cls in self.classes
        #                        for met in self._metrics}
        self._metric_values = defaultdict(list)
        self.img_reader = data_ops.ImageReader()

        self.merge_tumor_to_liver = kwargs.get("merge_tumor_to_liver", True)
        self.largest = kwargs.get("largest", True)
        self._timer = Timer()

    @property
    def classes(self):
        return self.params["model_instances"][0].classes[1:]  # Remove background

    @property
    def metrics(self):
        return self._metrics[:]

    @property
    def metric_values(self):
        return self._metric_values

    def append_metrics(self, pairs):
        for key, value in pairs.items():
            # if key in self._metric_values:
            self._metric_values[key].append(value)

    def clear_metrics(self):
        for key in self._metric_values:
            self._metric_values[key].clear()

    @staticmethod
    def maybe_append(dst, src, name, clear=False):
        if clear:
            dst.clear()
        if name in src:
            dst.append(src[name])

    def _evaluate(self, predicts, cases=None, verbose=False, save=False):
        # process a 3D image slice by slice or patch by patch
        logits3d = defaultdict(list)
        labels3d = defaultdict(list)
        # bg_masks3d = list()
        # cls_acc = list()
        # cls_pos = list()
        # cls_rec = list()
        self.clear_metrics()

        pad = -1
        bbox = None
        cur_case = None
        global_step = None

        self._timer.reset()
        self._timer.tic()
        for pred in predicts:
            new_case = pred["Names"][0].decode("utf-8")     # decode bytes string
            cur_case = cur_case or new_case
            pad = pad if pad != -1 else pred["Pads"][0]
            if "Bboxes" in pred:
                bbox = bbox if bbox is not None else pred["Bboxes"][0]
            if "GlobalStep" in pred:
                global_step = global_step if global_step is not None else pred["GlobalStep"]

            if cur_case == new_case:
                # Append batch to collections
                for c, cls in enumerate(self.classes):
                    logits3d[cls].append(np.squeeze(pred[cls + "Pred"], axis=-1))
                    labels3d[cls].append(pred["Labels_{}".format(c)])
                # self.maybe_append(bg_masks3d, pred, "BgMasks")
                # self.maybe_append(cls_acc, pred, "ClsAcc")
                # self.maybe_append(cls_pos, pred, "ClsPos")
                # self.maybe_append(cls_rec, pred, "ClsRec")
            elif cur_case + ".rev" == new_case:
                # Append reversed batch to another collections
                for c, cls in enumerate(self.classes):
                    logits3d[cls + "_rev"].append(np.squeeze(pred[cls + "Pred"], axis=-1))
            else:
                result = self._evaluate_case(logits3d, labels3d, cur_case, pad, bbox, save)
                # result = self._evaluate_case(logits3d, labels3d, cur_case, pad, bbox, save,
                #                              bg_masks3d=bg_masks3d, cls_acc=cls_acc, cls_pos=cls_pos,
                #                              cls_rec=cls_rec)
                self._timer.toc()
                if verbose:
                    log_str = "Evaluate-{} {}".format(self._timer.calls, Path(cur_case).stem)
                    for key, value in result.items():
                        log_str += " {}: {:.3f}".format(key, value)
                    tf.logging.info(log_str + " ({:.3f} secs/case)".format(self._timer.average_time))
                for c, cls in enumerate(self.classes):
                    logits3d[cls].clear()
                    labels3d[cls].clear()
                    logits3d[cls].append(np.squeeze(pred[cls + "Pred"], axis=-1))
                    labels3d[cls].append(pred["Labels_{}".format(c)])
                    if cls + "_rev" in logits3d:
                        logits3d[cls + "_rev"].clear()

                # self.maybe_append(bg_masks3d, pred, "BgMasks", clear=True)
                # self.maybe_append(cls_acc, pred, "ClsAcc", clear=True)
                # self.maybe_append(cls_pos, pred, "ClsPos", clear=True)
                # self.maybe_append(cls_rec, pred, "ClsRec", clear=True)

                if cases is not None and self._timer.calls >= cases:
                    break

                # Reset
                cur_case = new_case
                pad = pred["Pads"][0]
                if "Bboxes" in pred:
                    bbox = pred["Bboxes"][0]
                self._timer.tic()

        if cases is None or (self._timer.calls < cases):
            # Final case
            result = self._evaluate_case(logits3d, labels3d, cur_case, pad, bbox, save)
            # result = self._evaluate_case(logits3d, labels3d, cur_case, pad, bbox, save,
            #                              bg_masks3d=bg_masks3d, cls_acc=cls_acc, cls_pos=cls_pos,
            #                              cls_rec=cls_rec)
            self._timer.toc()
            if verbose:
                log_str = "Evaluate-{} {}".format(self._timer.calls, Path(cur_case).stem)
                for key, value in result.items():
                    log_str += " {}: {:.3f}".format(key, value)
                # log_str += " {}".format(list(bbox))
                tf.logging.info(log_str + " ({:.3f} secs/case)".format(self._timer.average_time))
        tf.logging.info("Evaluate all the dataset ({} cases) finished! ({:.3f} secs/case)"
                        .format(self._timer.calls, self._timer.average_time))

        # Compute average metrics
        results = {key: np.mean(values) for key, values in self._metric_values.items()}
        display_str = ""
        for key, value in results.items():
            display_str += "{}: {:.3f} ".format(key, value)
        tf.logging.info(display_str)

        if global_step is not None:
            results[tf.GraphKeys.GLOBAL_STEP] = global_step
        return results

    def evaluate_with_session(self, session=None, cases=None):
        predicts = [x for x in self._predict_keys
                    if x not in self.params["model_instances"][0].metrics_dict]
        # TODO(ZJW): Remove metrics_eval
        # predicts.extend(self.params["model_instances"][0].metrics_eval)
        predict_gen = self.model.predict_with_session(session, predicts, yield_single_examples=False)
        tf.logging.info("Begin evaluating 3D ...")
        return self._evaluate(predict_gen, cases=cases, verbose=True)

    def evaluate(self,
                 input_fn,
                 predict_keys=None,
                 hooks=None,
                 checkpoint_path=None,
                 cases=None):
        if not self.params["args"].use_fewer_guide:
            predict_gen = self.model.predict(input_fn, predict_keys, hooks, checkpoint_path,
                                             yield_single_examples=False)
            tf.logging.info("Begin evaluating ...")
            return self._evaluate(predict_gen, cases=cases, verbose=True,
                                  save=self.params["args"].save_predict)
        else:
            # Construct model with batch_size = 1
            # Disconnect input pipeline with model pipeline
            self.params["args"].batch_size = 1
            predict_gen = self.model.predict_with_guide(input_fn, predict_keys, hooks,
                                                        checkpoint_path, yield_single_examples=False)
            tf.logging.info("Begin evaluating ...")
            return self._evaluate(predict_gen, cases=cases, verbose=True,
                                  save=self.params["args"].save_predict)

    @staticmethod
    def _check_shapes_equal(volume_dict):
        # Don't use! Just for debug
        ref = {}
        mismatch = {}
        shape = None
        for key, value in volume_dict.items():
            if shape is None:
                shape = value.shape
                ref[key] = shape
            elif value.shape != shape:
                mismatch[key] = value.shape
        if len(mismatch) > 0:
            log_str = "Shape mismatch: Ref({} -> {}), Wrong(".format(*list(ref.items())[0])
            for key, value in mismatch.items():
                log_str += "{} -> {}  ".format(key, value)
            raise ValueError(log_str)

    def _evaluate_case(self, logits3d, labels3d, cur_case, pad, bbox=None, save=False):
        # Process a complete volume
        logits3d = {cls: np.concatenate(values) for cls, values in logits3d.items()}
        labels3d = {cls: np.concatenate(values) for cls, values in labels3d.items()}

        if pad != 0:
            logits3d = {cls: value[:-pad] if not cls.endswith("_rev") else value[pad:]
                        for cls, value in logits3d.items()}
            labels3d = {cls: value[:-pad] for cls, value in labels3d.items()}

        # For reversed volume
        keys = [x for x in logits3d.keys() if not x.endswith("_rev")]
        for key in keys:
            if (key + "_rev") in logits3d:
                # Merge two volumes which generated from different directions
                logits3d[key] = np.maximum(logits3d[key], np.flip(logits3d[key + "_rev"], axis=0))

        # livers3d = None
        # if kwargs.get("bg_masks3d", []):
        #     livers3d = (np.concatenate(kwargs["bg_masks3d"])[:-pad]
        #                 if pad != 0 else np.concatenate(kwargs["bg_masks3d"]))

        # Find volume voxel spacing from data source
        header = self.img_reader.header(cur_case)

        if bbox is not None:
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

        # Remove false positives outside liver region
        # if livers3d is not None and "Tumor" in logits3d:
        #     logits3d["Tumor"] *= livers3d.astype(logits3d["Tumor"].dtype)

        # Calculate 3D metrics
        cur_pairs = {}
        for c, cls in enumerate(self.classes):
            pairs = metric_ops.metric_3d(logits3d[cls], labels3d[cls],
                                         sampling=header.spacing, required=self.metrics)
            for met, value in pairs.items():
                cur_pairs["{}/{}".format(cls, met)] = value

        self.append_metrics(cur_pairs)

        if save:
            cur_case = Path(cur_case)
            case_name = cur_case.name.replace("volume", "prediction") + ".gz"
            save_path = Path(self.model.model_dir) / "prediction"
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
            pad_with = tuple(zip(bbox[2::-1], np.array(header.shape) - bbox[:2:-1] - 1))
            img_array = np.pad(img_array, pad_with, mode="constant", constant_values=0)

            self.img_reader.save(save_path, img_array, fmt=cur_case.suffix[1:])
            tf.logging.info("    ==> Save to {}"
                            .format(str(save_path.relative_to(save_path.parent.parent.parent))))
        return cur_pairs

    def compare(self, *args, **kwargs):
        return _compare(*args, **kwargs)


class EvaluateSlice(EvaluateBase):
    """ Evaluate Estimator model by slice """

    def __init__(self, model, **kwargs):
        """
        Parameters
        ----------
        model: CustomEstimator
            CustomEstimator instance
        kwargs: dict
            * merge_tumor_to_liver: bool, if `Tumor` and `Liver` in predictions (default True)
            * largest: bool, get largest component for liver, if `Liver` in predictions (default True)
        """
        if not isinstance(model, CustomEstimator):
            raise TypeError("model need a custom_estimator.CustomEstimator instance")
        super(EvaluateSlice, self).__init__(model, **kwargs)

        self.model = model
        # self._metric_values = {"{}/{}".format(cls, met): []
        #                        for cls in self.classes
        #                        for met in self._metrics}
        # if self.params["args"].cls_branch and "ClsAcc" in self._predict_keys:
        #     self._metric_values["ClsAcc"] = []
        self._metric_values = defaultdict(list)

    @property
    def classes(self):
        return self.params["model_instances"][0].classes[1:]  # Remove background

    @property
    def metric_values(self):
        return self._metric_values

    def append_metrics(self, pairs):
        for key, value in pairs.items():
            # if key in self._metric_values:
            self._metric_values[key].append(value)

    def clear_metrics(self):
        for key in self._metric_values:
            self._metric_values[key].clear()

    def evaluate_with_session(self, session=None, cases=None):
        predicts = list(self.params["model_instances"][0].metrics_dict.keys())
        tf.logging.info("Begin evaluating 2D ...")
        predict_gen = self.model.predict_with_session(session, predicts, yield_single_examples=False)

        self.clear_metrics()
        for pred in predict_gen:
            self.append_metrics(pred)

        results = {key: np.mean(values) for key, values in self._metric_values.items()}
        display_str = ""
        for key, value in results.items():
            display_str += "{}: {:.3f} ".format(key, value)
        tf.logging.info(display_str)

        return results

    def compare(self, *args, **kwargs):
        return _compare(*args, **kwargs)


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
