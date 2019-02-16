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


class EvaluateVolume(object):
    """ Evaluate Estimator model by volume """
    def __init__(self, model, metrics, predict_keys=None, **kwargs):
        """
        Parameters
        ----------
        model: tf.estimator.Estimator
            Estimator instance
        metrics: list of str
            3D metrics
        predict_keys: list of str
            Passed to self.model.predict
        kwargs: dict
            * merge_tumor_to_liver: bool, if `Tumor` and `Liver` in predictions (default True)
            * largest: bool, get largest component for liver, if `Liver` in predictions (default True)
        """
        if not isinstance(model, tf.estimator.Estimator):
            raise TypeError("model need a tf.estimator.Estimator instance")

        self.model = model
        self.params = model.params
        self._metrics = metrics
        self._predict_keys = predict_keys

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

    def evaluate(self, input_fn,
                 hooks=None,
                 checkpoint_path=None):
        """
        Evaluate model by combining 2D slices to 3D volume.

        Parameters
        ----------
        input_fn: callable
            Passes to Estimator.predict
        hooks: list of SessionRunHook
            Passes to Estimator.predict
        checkpoint_path: str
            Passes to Estimator.predict

        Returns
        -------

        """
        predicts = self.model.predict(input_fn, self._predict_keys, hooks,
                                      checkpoint_path, yield_single_examples=False)

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
            pairs = {"{}/{}".format(cls, met) for met, value in pairs.items()}
            self.append_metrics(pairs)
