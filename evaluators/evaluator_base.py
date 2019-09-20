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

import copy
from functools import reduce
from pathlib import Path
from collections import defaultdict


class EvaluateBase(object):
    """ Base class for custom evaluator """

    _metric_values = defaultdict(list)

    @property
    def metric_values(self):
        return self._metric_values

    def append_metrics(self, pairs):
        for key, value in pairs.items():
            self._metric_values[key].append(value)

    def clear_metrics(self):
        for key in self._metric_values:
            self._metric_values[key].clear()

    def save_metrics(self, save_file, save_dir=None):
        max_len = reduce(max, [len(val) for val in self._metric_values.values()])
        temp_metrics = copy.deepcopy(self._metric_values)
        for key in self._metric_values:
            diff = max_len - len(self._metric_values)
            temp_metrics[key].extend(["--"] * diff)

        keys = list(temp_metrics.keys())
        save_path = Path(save_dir) / save_file if save_dir else save_file
        with Path(save_path).open("w") as f:
            f.write(",".join(map(str, keys)) + "\n")
            for i in range(max_len):
                f.write(",".join([str(temp_metrics[key][i]) for key in keys]) + "\n")
        print("Write all metrics to", str(save_file))

    def run_with_session(self, session):
        """
        Evaluate model.

        Parameters
        ----------
        session: Session
            A Session to run evaluate(Please set in train mode).
            This parameter is useful for evaluation in training mode.

        Returns
        -------
        A dict with evaluation results

        """
        pass

    def run(self,
            input_fn,
            predict_keys=None,
            hooks=None,
            checkpoint_path=None):
        """
        Evaluate model.

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

    def compare(self,
                cur_result,
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
