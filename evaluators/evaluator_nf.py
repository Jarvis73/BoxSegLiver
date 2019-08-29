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


def get_evaluator(evaluator, estimator=None, model_dir=None, params=None):
    if evaluator == "Volume":
        return EvaluateVolume(estimator,
                              model_dir=model_dir,
                              params=params)
    # elif evaluator == "Slice":
    #     return EvaluateSlice(estimator)
    else:
        raise ValueError("Unsupported evaluator: {}. Must be [Volume, ]".format(evaluator))


class EvaluateVolume(EvaluateBase):
    """ Evaluate Estimator model by volume

    This class is for nf tumor segmentation.

    `estimator` used in run_with_session()
    `model_dir` used in run()
    """
    def __init__(self, estimator=None, model_dir=None, params=None):
        self.estimator = estimator
        self.model_dir = model_dir or estimator.model_dir
        self.params = params or estimator.params
        self.config = self.params["args"]
        self._timer = timer.Timer()
        meta_file = Path(__file__).parent.parent / "DataLoader/NF/prepare/meta.json"
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

        if hasattr(self.config, "use_spatial") and self.config.use_spatial:
            tf.logging.info("Enable --> use spatial guide")
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
            raise NotImplementedError
            # return self._run_with_session_actual_3d(session)
        else:
            return self._run_with_session_actual_2d(session)

    # def _run_with_session_actual_3d(self, session=None):
    #     data_list = input_pipeline_g._collect_datasets(
    #         self.config.test_fold, "eval_online",
    #         filter_tumor_size=self.config.filter_size,
    #         filter_only_liver_in_val=self.params.get("filter_only_liver_in_val", True))
    #     data_dict = {str(case["PID"]): case for case in data_list}
    #     # We just use model_instances[0] to get prediction name.
    #     # Its value will still be collected from multi-gpus
    #     predicts = ["labels", "names"] + list(self.params["model_instances"][0].predictions)
    #     predict_gen = self.estimator.evaluate_online(session, predicts, yield_single_examples=False)
    #     cur_case = None
    #     self._timer.reset()
    #     self.clear_metrics()
    #
    #     if not self.config.use_global_dice:
    #         tf.logging.info("Begin evaluating 3d at epoch end ...")
    #         self._timer.tic()
    #         volume_collection = defaultdict(list)
    #         for x in predict_gen:
    #             new_case = str(x["names"][0])
    #             cur_case = cur_case or new_case
    #             if cur_case == new_case:    # Append batch to collections
    #                 for cls in self.classes:
    #                     volume_collection[cls].append(np.squeeze(x[cls + "Pred"], axis=-1))
    #                 volume_collection["labels"].append(x["labels"])
    #             else:
    #                 volume_collection = {k: np.concatenate(v, axis=0) for k, v in volume_collection.items()}
    #                 z1, y1, x1, z2, y2, x2 = data_dict[cur_case]["bbox"]
    #                 pads = (self.config.batch_size - ((z2 - z1) % self.config.batch_size)) % self.config.batch_size
    #                 if pads > 0:
    #                     volume_collection = {k: v[:-pads] for k, v in volume_collection.items()}
    #                 volume = volume_collection
    #                 labels = volume_collection["labels"]
    #                 # Here we omit postprocess for saving training time
    #                 # volume = self._postprocess(volume, ori_shape=[y2 - y1, x2 - x1])
    #                 # labels = self._postprocess(labels, is_label=True, ori_shape=[y2 - y1, x2 - x1])
    #                 results = {}
    #                 for c, cls in enumerate(self.classes):
    #                     pairs = metric_ops.metric_3d(volume[cls], labels[cls], required=self.metrics_str)
    #                     for met, value in pairs.items():
    #                         results["{}/{}".format(cls, met)] = value
    #                 self.append_metrics(results)
    #                 volume_collection.clear()
    #                 for cls in self.classes:
    #                     volume_collection[cls].append(np.squeeze(x[cls + "Pred"], axis=-1))
    #                 volume_collection["labels"].append(x["labels"])
    #                 cur_case = new_case
    #             self._timer.toc()
    #             self._timer.tic()
    #
    #         volume_collection = {k: np.concatenate(v, axis=0) for k, v in volume_collection.items()}
    #         z1, y1, x1, z2, y2, x2 = data_dict[cur_case]["bbox"]
    #         pads = (self.config.batch_size - ((z2 - z1) % self.config.batch_size)) % self.config.batch_size
    #         if pads > 0:
    #             volume_collection = {k: v[:-pads] for k, v in volume_collection.items()}
    #         volume = volume_collection
    #         labels = volume["labels"]
    #         results = {}
    #         for c, cls in enumerate(self.classes):
    #             pairs = metric_ops.metric_3d(volume[cls], labels[cls], required=self.metrics_str)
    #             for met, value in pairs.items():
    #                 results["{}/{}".format(cls, met)] = value
    #         self.append_metrics(results)
    #
    #         display_str = "----Evaluate {} batches ".format(self._timer.calls)
    #         results = {key: np.mean(values) for key, values in self._metric_values.items()}
    #     else:
    #         tf.logging.info("Begin evaluating 3d at epoch end(global dice) ...")
    #         accumulator = defaultdict(int)
    #         self._timer.tic()
    #         volume_collection = defaultdict(list)
    #         for x in predict_gen:
    #             new_case = str(x["names"][0])
    #             cur_case = cur_case or new_case
    #             if cur_case == new_case:  # Append batch to collections
    #                 for cls in self.classes:
    #                     volume_collection[cls].append(np.squeeze(x[cls + "Pred"], axis=-1))
    #                 volume_collection["labels"].append(x["labels"])
    #             else:
    #                 volume_collection = {k: np.concatenate(v, axis=0) for k, v in volume_collection.items()}
    #                 z1, y1, x1, z2, y2, x2 = data_dict[cur_case]["bbox"]
    #                 pads = (self.config.batch_size - ((z2 - z1) % self.config.batch_size)) % self.config.batch_size
    #                 if pads > 0:
    #                     volume_collection = {k: v[:-pads] for k, v in volume_collection.items()}
    #                 volume = volume_collection
    #                 labels = volume_collection["labels"]
    #                 # Here we omit postprocess for saving training time
    #                 # volume = self._postprocess(volume_collection, ori_shape=[y2 - y1, x2 - x1])
    #                 # labels = self._postprocess(volume_collection["labels"], is_label=True,
    #                 #                            ori_shape=[y2 - y1, x2 - x1])
    #                 for i, cls in enumerate(self.classes):
    #                     conf = metric_ops.ConfusionMatrix(volume[cls].astype(int), (labels == i + 1).astype(int))
    #                     conf.compute()
    #                     accumulator[cls + "_fn"] += conf.fn
    #                     accumulator[cls + "_fp"] += conf.fp
    #                     accumulator[cls + "_tp"] += conf.tp
    #                 volume_collection = defaultdict(list)
    #                 for cls in self.classes:
    #                     volume_collection[cls].append(np.squeeze(x[cls + "Pred"], axis=-1))
    #                 volume_collection["labels"].append(x["labels"])
    #                 cur_case = new_case
    #
    #             self._timer.toc()
    #             self._timer.tic()
    #
    #         display_str = "----Evaluate {} batches ".format(self._timer.calls)
    #         results = {cls + "/Dice": 2 * accumulator[cls + "_tp"] / (
    #                 2 * accumulator[cls + "_tp"] + accumulator[cls + "_fn"] + accumulator[cls + "_fp"])
    #                    for cls in self.classes}
    #
    #     for key, value in results.items():
    #         display_str += "- {}: {:.3f} ".format(key, value)
    #     tf.logging.info(display_str + "({:.3f} secs)".format(self._timer.total_time))
    #     return results

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
            tf.logging.info("Begin evaluating 2d at epoch end(global dice) ...")
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
                # TODO(need): Need replace by other methods
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
                segmentation, seg_path, pads, bbox = labels
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
                    cur_case, seg_path, volume, bbox, dtype, save_path)

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
                model(model_inputs, ModeKeys.EVAL, *model_args, **model_kwargs)
                saver = tf.train.Saver()
                sess = tf.Session()
                # load weights
                sess.run(tf.global_variables_initializer())
                saver.restore(sess, checkpoint_path)

                predictions = {"Prob": model.probability, "TumorPred": model.predictions["TumorPred"]}

                if self.config.use_spatial:
                    eil = input_fn(ModeKeys.EVAL, self.params)  # EvalImage3DLoader
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
                    for features, labels in input_fn(ModeKeys.EVAL, self.params):
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

    def maybe_save_case(self, cur_case, vol_path, volume, dtype, save_path):
        if save_path:
            case_header = nib.load(str(vol_path)).header
            if dtype == "pred":
                volume = self._postprocess(volume)
                if "NF" in volume:
                    img_array = volume["NF"]
                else:
                    raise ValueError("Not supported save object!")
                save_file = save_path / "predict-{}.nii.gz".format(cur_case)
                nii_kits.write_nii(img_array, case_header, save_file)
            else:
                # Save 4D Tensor to npz
                img_array = volume
                save_file = save_path / (cur_case + ".npz")
                np.savez_compressed(str(save_file), img_array)
            tf.logging.info("    ==> Save to {}"
                            .format(str(save_file.relative_to(save_path.parent.parent.parent))))
            return volume, dtype == "pred"  # post_processed
        else:
            return volume, False

    def compare(self, *args_, **kwargs):
        return _compare(*args_, **kwargs)


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
