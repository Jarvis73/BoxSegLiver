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


class EvaluateBase(object):
    """ Base class for custom evaluator """

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

        self.model = model
        self.params = model.params
        self._metrics = model.params["args"].metrics_eval
        self._predict_keys = self.model._predict_keys

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
                 cases=None):
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
