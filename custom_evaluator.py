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
from collections import defaultdict

import loss_metrics as metric_ops
import utils.array_kits as arr_ops
import data_kits.build_data as data_ops
from custom_estimator import CustomEstimator


def add_arguments(parser):
    group = parser.add_argument_group(title="Evaluation Arguments")
    group.add_argument("--metrics_eval",
                       type=str,
                       default=["Dice"],
                       choices=["Dice", "VOE", "RVD", "ASSD", "RMSD", "MSD"],
                       nargs="+",
                       required=False, help="Evaluation metric names (default: %(default)s)")
    group.add_argument("--save_best",
                       action="store_false",
                       required=False, help="Save checkpoint with best results")
    group.add_argument("--eval_steps",
                       type=int,
                       default=2500,
                       required=False, help="Evaluate pre several steps (default: %(default)d)")
    group.add_argument("--primary_metric",
                       type=str,
                       required=False, help="Primary metric for evaluation. Typically it has format "
                                            "<class>/<metric>")
    group.add_argument("--secondary_metric",
                       type=str,
                       required=False, help="Secondary metric for evaluation. Typically it has format "
                                            "<class>/<metric>")


def get_eval_params(eval_steps=2500,
                    largest=True,
                    merge_tumor_to_liver=False,
                    primary_metric=None,
                    secondary_metric=None):
    return {
        "evaluator": EvaluateVolume,
        "eval_steps": eval_steps,
        "eval_kwargs": {"merge_tumor_to_liver": merge_tumor_to_liver, "largest": largest},
        "primary_metric": primary_metric,
        "secondary_metric": secondary_metric
    }


class EvaluateBase(object):
    """ Base class for custom evaluator """

    def __init__(self, model, predict_keys=None, **kwargs):
        """
        Parameters
        ----------
        model: CustomEstimator
            CustomEstimator instance
        predict_keys: list of str
            Passed to self.model.predict
        kwargs: dict
            * merge_tumor_to_liver: bool, if `Tumor` and `Liver` in predictions (default True)
            * largest: bool, get largest component for liver, if `Liver` in predictions (default True)
        """

        self.model = model
        self.params = model.params
        self._metrics = model.params["args"].metrics_eval
        self._predict_keys = predict_keys

    def evaluate_with_session(self, session):
        """
        Evaluate model by combining 2D slices to 3D volume.

        Parameters
        ----------
        session: Session
            A Session to run evaluate(Please set in train mode).
            This parameter is useful for evaluation in training mode.

        Returns
        -------
        A dict with
            * evaluation results
            * global_step

        """
        pass

    def evaluate(self,
                 input_fn,
                 predict_keys=None,
                 hooks=None,
                 checkpoint_path=None,
                 yield_single_examples=False):
        """
        Evaluate model by combining 2D slices to 3D volume.

        Parameters
        ----------
        All parameters are passed into predict function

        Returns
        -------
        A dict with
            * evaluation results
            * global step

        """
        pass

    @staticmethod
    def compare(cur_result,
                ori_result,
                primary_metric=None,
                secondary_metric=None):
        """
        Comparision function for two results

        Parameters
        ----------
        cur_result: dict
            current result
        ori_result: dict
            origin best result
        primary_metric: str
            primary metric, dict key
        secondary_metric: str
            secondary metric, dict key

        Returns
        -------
        True for cur_result better than ori_result
        False for cur_result not better than ori_result

        """
        pass


class EvaluateVolume(EvaluateBase):
    """ Evaluate Estimator model by volume """
    def __init__(self, model, predict_keys=None, **kwargs):
        """
        Parameters
        ----------
        model: CustomEstimator
            CustomEstimator instance
        predict_keys: list of str
            Passed to self.model.predict
        kwargs: dict
            * merge_tumor_to_liver: bool, if `Tumor` and `Liver` in predictions (default True)
            * largest: bool, get largest component for liver, if `Liver` in predictions (default True)
        """
        if not isinstance(model, CustomEstimator):
            raise TypeError("model need a custom_estimator.CustomEstimator instance")
        super(EvaluateVolume, self).__init__(model, predict_keys, **kwargs)

        self._metric_values = {"{}/{}".format(cls, met): []
                               for cls in self.classes
                               for met in self._metrics}
        self.img_reader = data_ops.ImageReader()

        self.merge_tumor_to_liver = kwargs.get("merge_tumor_to_liver", True)
        self.largest = kwargs.get("largest", True)

    @property
    def classes(self):
        return self.params["model"].classes[1:]  # Remove background

    @property
    def metrics(self):
        return self._metrics[:]

    @property
    def metric_values(self):
        return self._metric_values

    def append_metrics(self, pairs):
        for key, value in pairs.items():
            if key in self._metric_values:
                self._metric_values[key].append(value)

    def _evaluate(self, predicts):
        # process a 3D image slice by slice or patch by patch
        logits3d = defaultdict(list)
        labels3d = defaultdict(list)

        pad = -1
        cur_case = None
        global_step = None
        for pred in predicts:
            global_step = global_step if global_step is not None else pred["GlobalStep"]
            cur_case = cur_case or pred["Names"][0]
            pad = pad if pad != -1 else pred["Pads"][0]
            if cur_case == pred["Names"][0]:
                # Append batch to collections
                for c, cls in enumerate(self.classes):
                    logits3d[cls].append(pred[cls + "Pred"])
                    labels3d[cls].append(pred["Labels"][c])
            else:
                self._evaluate_case(logits3d, labels3d, cur_case, pad)
                for cls in self.classes:
                    logits3d[cls].clear()
                    labels3d[cls].clear()

                # Reset
                cur_case = pred["Names"][0]
                pad = pred["Pads"][0]

        # Final case
        self._evaluate_case(logits3d, labels3d, cur_case, pad)

        # Compute average metrics
        results = {key: np.mean(values) for key, values in self._metric_values}
        results[tf.GraphKeys.GLOBAL_STEP] = global_step
        return results

    def evaluate_with_session(self, session=None):
        predicts = self.model.predict_with_session(session, yield_single_examples=False)
        return self._evaluate(predicts)

    def evaluate(self,
                 input_fn,
                 predict_keys=None,
                 hooks=None,
                 checkpoint_path=None,
                 yield_single_examples=False):
        predicts = self.model.predict(input_fn, predict_keys, hooks, checkpoint_path, yield_single_examples)
        return self._evaluate(predicts)

    def _evaluate_case(self, logits3d, labels3d, cur_case, pad):
        # Process a complete volume
        logits3d = {cls: np.concatenate(values)[:-pad] if pad != 0 else np.concatenate(values)
                    for cls, values in logits3d.items()}
        labels3d = {cls: np.concatenate(values)[:-pad] if pad != 0 else np.concatenate(values)
                    for cls, values in labels3d.items()}

        # Add tumor to liver volume
        if self.merge_tumor_to_liver and "Tumor" in logits3d and "Liver" in logits3d:
            logits3d["Liver"] += logits3d["Tumor"]
            labels3d["Liver"] += labels3d["Tumor"]

        # Find largest component --> for liver
        if self.largest and "Liver" in logits3d:
            logits3d["Liver"] = arr_ops.get_largest_component(logits3d["Liver"], rank=3)

        # Find volume voxel spacing from data source
        spacing = self.img_reader.header(cur_case).spacing()

        # Calculate 3D metrics
        for c, cls in enumerate(self.classes):
            pairs = metric_ops.metric_3d(logits3d[cls], labels3d[cls],
                                         sampling=spacing, required=self.metrics)
            pairs = {"{}/{}".format(cls, met): value for met, value in pairs.items()}
            self.append_metrics(pairs)

    @staticmethod
    def compare(cur_result,
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
            raise KeyError("`primary_metric` not in valid result key")
        if secondary_metric and secondary_metric not in cur_result:
            raise KeyError("`secondary_metric` not in valid result key")
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
