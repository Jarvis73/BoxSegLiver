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
                    use_sg_reduce_fp=False,
                    primary_metric=None,
                    secondary_metric=None):
    if evaluator not in ["Volume", "Slice"]:
        raise ValueError("Unsupported evaluator: {}. Must be [Volume, Slice]".format(evaluator))
    return {
        "evaluator": eval("Evaluate" + evaluator),
        "eval_steps": eval_steps,
        "eval_kwargs": {"merge_tumor_to_liver": merge_tumor_to_liver,
                        "largest": largest,
                        "use_sg_reduce_fp": use_sg_reduce_fp},
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
        # if not isinstance(model, CustomEstimator):
        #     raise TypeError("model need a custom_estimator.CustomEstimator instance")
        super(EvaluateVolume, self).__init__(model, **kwargs)

        self.img_reader = data_ops.ImageReader()

        self.merge_tumor_to_liver = kwargs.get("merge_tumor_to_liver", True)
        self.largest = kwargs.get("largest", True)
        self.use_sg_reduce_fp = kwargs.get("use_sg_reduce_fp", False)
        if self.merge_tumor_to_liver:
            tf.logging.info("Enable --> merge_tumor_to_liver")
        if self.largest:
            tf.logging.info("Enable --> largest")
        if self.use_sg_reduce_fp:
            tf.logging.info("Enable --> use_sg_reduce_fp")
        self._timer = Timer()

    @property
    def classes(self):
        return self.params["model_instances"][0].classes[1:]  # Remove background

    @property
    def metrics_str(self):
        return self.model.params["args"].metrics_eval

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
        self.clear_metrics()

        pad = -1
        bbox = None
        cur_case = None
        global_step = None

        self._timer.reset()
        self._timer.tic()
        for predict in predicts:
            new_case = predict["Names"][0].decode("utf-8")     # decode bytes string
            cur_case = cur_case or new_case
            pad = pad if pad != -1 else predict["Pads"][0]
            if "Bboxes" in predict:
                bbox = bbox if bbox is not None else predict["Bboxes"][0]
            if "GlobalStep" in predict:
                global_step = global_step if global_step is not None else predict["GlobalStep"]

            if cur_case == new_case:
                # Append batch to collections
                for c, cls in enumerate(self.classes):
                    logits3d[cls].append(np.squeeze(predict[cls + "Pred"], axis=-1))
                    labels3d[cls].append(predict["Labels_{}".format(c)])
                # self.maybe_append(labels3d["SpGuide"], pred, "SpGuide")
                # self.maybe_append(bg_masks3d, pred, "BgMasks")
            elif cur_case + ".rev" == new_case:
                # Append reversed batch to another collections
                for c, cls in enumerate(self.classes):
                    logits3d[cls + "_rev"].append(np.squeeze(predict[cls + "Pred"], axis=-1))
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
                    logits3d[cls].append(np.squeeze(predict[cls + "Pred"], axis=-1))
                    labels3d[cls].append(predict["Labels_{}".format(c)])
                    if cls + "_rev" in logits3d:
                        logits3d[cls + "_rev"].clear()

                # self.maybe_append(labels3d["SpGuide"], pred, "SpGuide", clear=True)
                # self.maybe_append(bg_masks3d, pred, "BgMasks", clear=True)

                if cases is not None and self._timer.calls >= cases:
                    break

                # Reset
                cur_case = new_case
                pad = predict["Pads"][0]
                if "Bboxes" in predict:
                    bbox = predict["Bboxes"][0]
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

        self.save_metrics("metrics_3d.txt", self.model.model_dir)
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

        # Remove false positives with spatial guide
        if self.use_sg_reduce_fp:
            logits3d["Tumor"] = arr_ops.reduce_fp_with_guide(np.squeeze(labels3d["Tumor"], axis=-1),
                                                             logits3d["Tumor"],
                                                             guide=self.params["args"].guide)

        # Calculate 3D metrics
        cur_pairs = {}
        for c, cls in enumerate(self.classes):
            pairs = metric_ops.metric_3d(logits3d[cls], labels3d[cls],
                                         sampling=header.spacing, required=self.metrics_str)
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

    def compare(self, *args_, **kwargs):
        return _compare(*args_, **kwargs)


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
        # if not isinstance(model, CustomEstimator):
        #     raise TypeError("model need a custom_estimator.CustomEstimator instance")
        super(EvaluateSlice, self).__init__(model, **kwargs)

    @property
    def classes(self):
        return self.params["model_instances"][0].classes[1:]  # Remove background

    def evaluate_with_session(self, session=None, cases=None):
        predicts = list(self.params["model_instances"][0].metrics_dict.keys())
        tf.logging.info("Begin evaluating 2D ...")
        predict_gen = self.model.predict_with_session(session, predicts, yield_single_examples=False)

        self.clear_metrics()
        for predict in predict_gen:
            self.append_metrics(predict)

        results = {key: np.mean(values) for key, values in self._metric_values.items()}
        display_str = ""
        for key, value in results.items():
            display_str += "{}: {:.3f} ".format(key, value)
        tf.logging.info(display_str)

        return results

    def evaluate(self,
                 input_fn,
                 predict_keys=None,
                 hooks=None,
                 checkpoint_path=None,
                 cases=None):
        predicts = ["Names", "Indices"]
        tf.logging.info("Begin evaluating 2D ...")
        if not self.params["args"].use_fewer_guide:
            predict_gen = self.model.predict(input_fn, predicts, hooks, checkpoint_path,
                                             yield_single_examples=True)
        else:
            # Construct model with batch_size = 1
            # Disconnect input pipeline with model pipeline
            self.params["args"].batch_size = 1
            predict_gen = self.model.predict_with_guide(input_fn, predicts, hooks,
                                                        checkpoint_path, yield_single_examples=True)

        self.clear_metrics()
        for i, pred_ in enumerate(predict_gen):
            print("\rEval {} examples ...".format(i + 1), end="")
            self.append_metrics(pred_)
        print()
        self.save_metrics("metrics_2d.txt", self.model.model_dir)

        results = {key: float(np.mean(values)) for key, values in self._metric_values.items()
                   if key not in ["Names", "Indices"]}
        display_str = ""
        for key, value in results.items():
            display_str += "{}: {:.3f} ".format(key, value)
        tf.logging.info(display_str)

        return results

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


class TumorManager(object):
    def __init__(self, tumor_info):
        self._tumor_info = tumor_info
        self._name = None
        self._bbox = None
        self._id = None
        self.direction = 1  # 1 for "forward", -1 for "backward"
        self.disc = ndi.generate_binary_structure(2, connectivity=1)
        self.guides = None
        self.pred = None
        self.backup = defaultdict(list)
        self.debug = False

    @property
    def info(self):
        return self._tumor_info

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, new_name):
        self._name = new_name
        self._name_id = int(Path(new_name).name.split(".")[0].split("-")[1])
        self.total_tumors = self._tumor_info[self._tumor_info["PID"] ==
                                             "segmentation-{}.nii".format(self._name_id)]
        self.total_tumors = self.total_tumors.iloc[:, range(2, 8)].values
        self.total_tumors[:, [3, 4, 5]] -= 1    # [) --> []
        self.total_tumors_yx = self.total_tumors[:, [1, 0, 4, 3]]  # [y1, x1, y2, x2]
        self.total_tumors_z = self.total_tumors[:, [2, 5]]          # range: [z1, z2]
        del self.total_tumors

    @property
    def bbox(self):
        return self._bbox

    def set_bbox(self, new_bbox, shape):
        """
        new_bbox: make sure (x1, y1, z1, x2, y2, z2)
        """
        self._bbox = np.asarray(new_bbox)   # []
        self.shape = np.asarray(shape)
        scale = self.shape / np.array([self._bbox[4] - self._bbox[1] + 1,
                                       self._bbox[3] - self._bbox[0] + 1])
        self.total_tumors_yx = (self.total_tumors_yx - self._bbox[[1, 0, 1, 0]]) * np.tile(scale, [2])
        self.total_tumors_z -= self._bbox[[2, 2]]

    @property
    def id(self):
        """ slice id in CT """
        return self._id

    def _set_id(self, new_id):
        if self._name is None or self._bbox is None:
            raise ValueError("Please set name and bbox first")
        self._id = new_id

    def clear_backup(self):
        for key in self.backup:
            self.backup[key].clear()

    def append_backup(self, center, stddev, zi):
        self.backup["centers"].append(center)
        self.backup["stddevs"].append(stddev)
        self.backup["zi"].append(zi)

    def reset(self, direction=1):
        self._name = None
        self._bbox = None
        self._id = None
        self.direction = direction
        self.guides = None
        self.pred = None
        self.clear_backup()

    def print(self, *args_, **kwargs):
        if self.debug:
            print(*args_, **kwargs)

    def set_guide_info(self, guide, new_id):
        """
        Decide whether to finish a tumor or not.
        Of course making this decision with only guide is enough.

        Parameters
        ----------
        guide: Tensor
            with shape [n, 4]
        new_id: int
            Index number of current slice in a CT volume.
        """
        self._set_id(new_id)

        select = np.where(guide[:, 0] >= 0)[0]
        self.centers_yx = np.round(guide[select, 1::-1] * self.shape).astype(np.int32)
        self.stddevs_yx = np.round(guide[select, :1:-1] * self.shape).astype(np.int32)
        # print(guide)
        # Indices for self.total_tumor_z
        self.zi = np.array([self.determine_z_min_max(i, ctr, std)
                            for i, (ctr, std) in enumerate(zip(self.centers_yx, self.stddevs_yx))],
                           dtype=np.int32)
        self.print("{} New Guide: {}".format(new_id + self._bbox[2], self.zi.shape[0]), end="")
        if self.backup["zi"]:
            # TODO(ZJW): Maybe we should remove the same objects from centers_ys and centers_backup.
            #            For example, two or more adjacent slices are provided guides.
            self.centers_yx = np.concatenate(
                (self.centers_yx, np.asarray(self.backup["centers"], np.int32)), axis=0)
            self.stddevs_yx = np.concatenate(
                (self.stddevs_yx, np.asarray(self.backup["stddevs"], np.int32)), axis=0)
            self.zi = np.concatenate(
                (self.zi, np.asarray(self.backup["zi"], np.int32)), axis=0)
        self.print("  Last Guide: {}".format(len(self.backup["zi"])))

    def get_guide_image(self, guide, new_id):
        self.set_guide_info(guide, new_id)

        if len(self.centers_yx) > 0:
            self.guides = arr_ops.create_gaussian_distribution(self.shape,
                                                               self.centers_yx[0, ::-1],
                                                               self.stddevs_yx[0, ::-1])
            for i in range(1, len(self.centers_yx)):
                self.guides = np.maximum(self.guides, arr_ops.create_gaussian_distribution(
                    self.shape, self.centers_yx[i, ::-1], self.stddevs_yx[i, ::-1]))
        else:
            self.guides = np.zeros(self.shape, dtype=np.float32)

        if self.pred is None:
            return self.guides[None, ..., None]
        else:
            self.guides = np.maximum(
                self.guides, arr_ops.get_gd_image_multi_objs(
                    self.pred, center_perturb=0., stddev_perturb=0.))
            return self.guides[None, ..., None]

    def check_pred(self, predict, filter_thresh=0.15):
        """
        Remove those predicted tumors who are out of range.
        Apply supervisor to predicted tumors.

        Make sure `pred` is binary

        self.pred which is created in this function will be used for generating next guide.
        So don't use `self.pred` as real prediction, because we will also remove those ended
        in current slice from self.pred.

        TODO(ZJW): adjust filter_thresh 0.35 ?
        """
        if self.guides is None:
            raise ValueError("previous_guide is None")
        if np.sum(predict) == 0:
            return predict

        self.clear_backup()

        labeled_objs, n_objs = ndi.label(predict, self.disc)
        slicers = ndi.find_objects(labeled_objs)
        # Decide whether reaching the end of the tumor or not
        for i, slicer in zip(range(n_objs), slicers):
            res_obj = labeled_objs == i + 1
            res_obj_slicer = res_obj[slicer]
            # 1. Filter wrong tumors(no corresponding guide)
            mask_guide_by_res = res_obj_slicer * self.guides[slicer]
            # print(np.max(mask_guide_by_res))
            if np.max(mask_guide_by_res) < filter_thresh:
                self.print("Remove")
                predict[slicer] -= res_obj_slicer   # Faster than labeled_objs[res_obj] = 0
                continue
            # 2. Match res_obj to guide
            res_peak_pos = list(np.unravel_index(mask_guide_by_res.argmax(), mask_guide_by_res.shape))
            res_peak_pos[0] += slicer[0].start
            res_peak_pos[1] += slicer[1].start
            #   2.1. Check whether res_peak is just a guide center
            found = -1
            for j, center in enumerate(self.centers_yx):
                if res_peak_pos[0] == center[0] and res_peak_pos[1] == center[1]:
                    found = j   # res_peak is just a center
                    break
            #   2.2. From the nearest guide center, check that whether it is the corresponding guide.
            #        Rule: Image(guide) values along the line from res_obj's peak to its corresponding
            #        guide center must be monotonously increasing.
            if found < 0:   # gradient ascent from res_peak to center
                # compute distances between res_obj_peak and every guide center
                distances = np.sum((self.centers_yx - res_peak_pos) ** 2, axis=1)
                order = np.argsort(distances)
                for j in order:
                    ctr = self.centers_yx[j]
                    if self.ascent_line(self.guides, res_peak_pos[1], res_peak_pos[0], ctr[1], ctr[0]):
                        # Found
                        found = j
                        break
            if found < 0:
                raise ValueError("Can not find corresponding guide!")
            # 3. Check z range and stop finished tumors(remove from next guide image)
            if (self.direction == 1 and self._id >= self.total_tumors_z[self.zi[found]][1]) or \
                    (self.direction == -1 and self._id <= self.total_tumors_z[self.zi[found]][0]):
                # if self.direction == -1:
                #     print("End {} vs {}, {}".format(self._id, self.total_tumors_z[self.zi[found]][0],
                #                                     self._bbox[2]))
                # if self.direction == 1:
                #     print("End {} vs {}, {}".format(self._id, self.total_tumors_z[self.zi[found]][1],
                #                                     self._bbox[2]))
                predict[slicer] -= res_obj_slicer
                continue
            # 4. Compute moments. Save moments of tumors for next slice
            ctr, std = arr_ops.compute_robust_moments(res_obj_slicer, index="ij")
            ctr[0] += slicer[0].start
            ctr[1] += slicer[1].start
            self.append_backup(ctr, std, self.zi[found])
            # print(ctr, std, self.zi[found])

        self.pred = predict

    @staticmethod
    def ascent_line(img, x0, y0, x1, y1):
        # Find points along this line
        xs, ys, forward = arr_ops.xiaolinwu_line(x0, y0, x1, y1)
        ascent = True
        pre = img[ys[0], xs[0]] if forward else img[ys[-1], xs[-1]]
        xs, ys = (xs, ys) if forward else (reversed(xs[:-1]), reversed(ys[:-1]))
        for x, y in zip(xs, ys):
            cur = img[y, x]
            if cur >= pre:
                pre = cur
                continue
            else:
                ascent = False
                break
        return ascent

    def determine_z_min_max(self, idx, center, stddev):
        _ = stddev  # Unused
        diff = self.total_tumors_yx - np.tile(center, [2])
        sign = np.all(diff[:, 0:2] * diff[:, 2:4] <= 0, axis=1)
        select = np.where(sign)[0]
        if len(select) == 0:
            nn_dist = np.mean(np.abs(diff), axis=1)
            nn_idx = np.argmin(nn_dist)
            if nn_dist[nn_idx] < 2.:
                return nn_idx
            else:
                print("#" * 50)
                print(nn_idx, nn_dist[nn_idx])
                print(self._id, idx, center)
                print(self._bbox)
                print(self.total_tumors_yx)
                print("#" * 50)
                raise ValueError
        elif len(select) == 1:
            return select[0]
        else:
            tumor_centers = (self.total_tumors_yx[select, 0:2] + self.total_tumors_yx[select, 2:4]) / 2
            distance = np.sum((tumor_centers - center) ** 2, axis=1)
            new_sel = np.argmin(distance)
            return select[new_sel]


if __name__ == "__main__":
    import pandas as pd
    import input_pipeline_osmn
    import matplotlib.pyplot as plt
    from utils import nii_kits

    tumor_path = Path(__file__).parent / "data/LiTS/tumor_summary.csv"
    tumors_info = pd.read_csv(str(tumor_path))

    class Foo(object):
        pass

    args = Foo()
    args.input_group = 3
    args.eval_skip_num = 20
    args.batch_size = 1
    args.num_gpus = 1
    args.guide = "middle"
    args.resize_for_batch = True
    args.im_height = 256
    args.im_width = 256
    args.hist_scale = 1.0
    args.w_width = 450
    args.w_level = 50

    n = 0
    dataset = input_pipeline_osmn.get_3d_multi_records_dataset_for_eval(
        [r"D:\documents\MLearning\MultiOrganDetection\core\MedicalImageSegmentation"
         r"\data\LiTS\records\trainval-bbox-3D-3-of-5.tfrecord"],
        [r"D:\documents\MLearning\MultiOrganDetection\core\MedicalImageSegmentation"
         r"\data\LiTS\records\hist-100--200_250-3D-3-of-5.tfrecord"],
        mode="eval",
        args=args
    ).skip(n).make_one_shot_iterator().get_next()

    def run(mgr):
        _, temp = nii_kits.nii_reader(Path(__file__).parent / "model_dir/016_osmn_in_noise"
                                                              "/prediction/prediction-113.nii.gz")
        temp = arr_ops.merge_labels(temp, [0, 2])
        temp = temp[arr_ops.bbox_to_slices(mgr.bbox)]
        temp = ndi.zoom(temp, [1, mgr.shape[0] / temp.shape[1], mgr.shape[1] / temp.shape[2]],
                        order=0)[n:]
        for x in np.concatenate((temp, np.flip(temp, axis=0)), axis=0):
            yield x

    sess = tf.Session()
    features_val, labels_val = sess.run(dataset)
    # print(features_val["names"], np.clip(labels_val - 1, 0, 1).sum(), features_val["sp_guide"])
    # features_val, labels_val = sess.run(dataset)
    # print(features_val["names"], np.clip(labels_val - 1, 0, 1).sum(), features_val["sp_guide"])

    t_mgr = TumorManager(tumors_info)
    t_mgr.name = features_val["names"][0].decode("utf-8")
    z0 = features_val["bboxes"][0][2]
    t_mgr.set_bbox(features_val["bboxes"][0], shape=features_val["images"].shape[1:-1])
    features_val["sp_guide"] = t_mgr.get_guide_image(features_val["sp_guide"][0], new_id=n)
    run_gen = run(t_mgr)

    fig, ax = plt.subplots(1, 2)
    init = next(run_gen)
    t_mgr.check_pred(init)
    init = init.copy()
    init[0, 0] = 1
    init2 = features_val["sp_guide"][0, ..., 0]
    init2[0, 0] = 1
    spg_handle = ax[0].imshow(init2, cmap="gray")
    img_handle = ax[1].imshow(init, cmap="gray")
    text = plt.title("{}".format(n + z0))
    n += 1

    cur_name = features_val["names"][0]

    def key_press_event(event):
        global n, features_val, labels_val, pred, cur_name
        if event.key == "down":
            features_val, labels_val = sess.run(dataset)
            new_name = features_val["names"][0]
            if new_name != cur_name:
                print(new_name.decode("utf-8"))
                cur_name = new_name
                n = n - 1
                t_mgr.reset(-1)
                t_mgr.name = cur_name.decode("utf-8")
                t_mgr.set_bbox(features_val["bboxes"][0], features_val["images"].shape[1:-1])
            features_val["sp_guide"] = t_mgr.get_guide_image(features_val["sp_guide"][0], new_id=n)
            pred = next(run_gen)
            t_mgr.check_pred(pred)
            spg_handle.set_data(features_val["sp_guide"][0, ..., 0])
            img_handle.set_data(pred)
            text.set_text("{}".format(n + z0))
            if t_mgr.direction == 1:
                n += 1
            else:
                n -= 1
            fig.canvas.draw()

    fig.canvas.mpl_connect("key_press_event", key_press_event)
    plt.show()
