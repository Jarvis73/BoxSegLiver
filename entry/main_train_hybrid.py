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

import cv2
import sys
import zlib
import math
import pickle
import argparse
from pathlib import Path
from collections import defaultdict
import numpy as np
import scipy.ndimage as ndi
from skimage.morphology import skeletonize_3d
import tensorflow as tf     # Tensorflow >= 1.13.0
from tensorflow.python.platform import tf_logging as logging
from skimage.segmentation import find_boundaries
from scipy.ndimage.morphology import distance_transform_edt

import config
import loss_metrics
from core import models
from core import solver
from DataLoader.NF import input_pipeline_3d as input_pipeline
from DataLoader import misc
from DataLoader.Liver import nii_kits
from utils.array_kits import create_gaussian_distribution_v2
from utils.ckpt_kits import find_checkpoint
from utils.logger import create_logger


def _get_arguments():
    parser = argparse.ArgumentParser()

    config.add_arguments(parser)
    models.add_arguments(parser)
    solver.add_arguments(parser)
    loss_metrics.add_arguments(parser)
    input_pipeline.add_arguments(parser)

    group = parser.add_argument_group(title="Hybrid Training Arguments")
    group.add_argument("--ckpt_2d", type=str)
    group.add_argument("--ckpt_3d", type=str)

    args = parser.parse_args()
    config.check_args(args, parser)
    config.fill_default_args(args)

    return args


def _custom_tf_logger(args):
    # Set custom logger
    log_dir = Path(args.model_dir) / "logs"
    if not log_dir.exists():
        log_dir.mkdir(parents=True, exist_ok=True)

    if args.out_file:
        log_file = log_dir / args.out_file
        with_time = False
    else:
        log_file = log_dir / "{}_{}".format(args.mode, args.tag)
        with_time = True
    logging._logger = create_logger(log_file=log_file, with_time=with_time, file_level=1,
                                    clear_exist_handlers=True, name="tensorflow")


def filter_tiny_nf(mask):
    struct2 = ndi.generate_binary_structure(2, 1)
    for i in range(mask.shape[0]):
        res, n_obj = ndi.label(mask[i], struct2)
        size = np.bincount(res.flat)
        for j in np.where(size <= 2)[0]:
            mask[i][res == j] = 0

    struct3 = ndi.generate_binary_structure(3, 2)
    res, n_obj = ndi.label(mask, struct3)
    size = np.bincount(res.flat)
    for i in np.where(size <= 5)[0]:
        mask[res == i] = 0
    return mask


def load_dataset(cfg, release_label=True):
    data = input_pipeline.load_data() if not cfg.downsampling else input_pipeline.load_data_ds()
    dataset = input_pipeline.load_split(cfg.test_fold, mode=cfg.infer_set)
    dataset = dataset[[True if item.pid in data and item.remove != 1 else False for _, item in dataset.iterrows()]]
    # dataset containing nf (remove benign scans)
    dataset['nf'] = [True if len(data[pid]['lab_rng']) > 1 else False for pid in dataset.pid]
    nf_set = dataset[dataset.nf]

    slim_labels_path = Path(__file__).parent.parent / "data/NF/slim_labels.gz.pkl"
    if slim_labels_path.exists():
        print(f"Loading slimed label cache from {slim_labels_path}")
        with slim_labels_path.open("rb") as f:
            slim_labels = pickle.loads(zlib.decompress(f.read()))
        for i in data:
            data[i]['slim'] = slim_labels[i]
        print("Finished!")
    else:
        slim_labels = {}
        print(f"Saving slimed label cache to {slim_labels_path}")
        for i, item in data.items():
            slim_labels[i] = filter_tiny_nf(np.clip(item['lab'], 0, 1).copy())
            data[i]['slim'] = slim_labels[i]
        with slim_labels_path.open("wb") as f:
            f.write(zlib.compress(pickle.dumps(slim_labels)))
        print("Finished!")

    if release_label:
        for i in data:
            data[i].pop('lab')

    if cfg.save_predict:
        save_path = Path(cfg.model_dir) / cfg.save_subdir
        save_path.mkdir(parents=True, exist_ok=True)
    else:
        save_path = None

    return data, nf_set, save_path


def inter_simulation_test(pred, ref):
    """
    Interaction simulation, including positive points and negative points in test time

    1. Compute false positive and false negative areas
    2. Connectivity analysis
    3. Choice the largest error region
    4. Please a new click on the center of the error region
    5. Determine foreground or background pixel

    Parameters
    ----------
    pred: np.ndarray, 3d, prediction
    ref:  np.ndarray, 3d, reference

    Returns
    -------
    pos: np.ndarray, a list of two values (y, x)
    fg: 0 for positive guide, 1 for negative
    """
    pred = pred.astype(np.bool)
    sym_diff = pred ^ ref
    struct = ndi.generate_binary_structure(3, 1)
    res, n_obj = ndi.label(sym_diff, struct)
    counts = np.bincount(res.flat)
    max_i = np.argmax(counts[1:]) + 1
    area = np.stack(np.where(res == max_i), axis=1)
    pos = np.mean(area, axis=0).round(0).astype(np.int32)
    if not sym_diff[pos[0], pos[1], pos[2]]:
        ske = np.stack(np.where(skeletonize_3d(sym_diff)), axis=1)
        min_i = np.argmin(np.sum((ske - pos) ** 2, axis=1))
        pos = ske[min_i]
    fg = 0 if ref[pos[0], pos[1], pos[2]] else 1
    return pos, fg


def _get_session_config(args):
    if args.device_mem_frac > 0:
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.device_mem_frac)
    else:
        gpu_options = tf.GPUOptions(allow_growth=True)
    sess_cfg = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)
    return sess_cfg


def build_graph(shape, cfg):
    with tf.Graph().as_default() as g:
        images = tf.placeholder(tf.float32, shape=shape, name="Image")
        inputs = {"images": images}

        if hasattr(cfg, "use_spatial") and cfg.use_spatial:
            sp_guide = tf.placeholder(tf.float32, shape=shape[:-1] + (cfg.guide_channel,), name="Guide")
            inputs["sp_guide"] = sp_guide

        model_params = models.get_model_params(cfg)
        model = model_params["model"](cfg)
        kwargs = model_params.get("model_kwargs", {})
        kwargs["ret_prob"] = True   # For TTA
        kwargs["ret_pred"] = False
        model(inputs, cfg.mode, *model_params.get("model_args", ()), **kwargs)

        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
    return inputs, model, init, saver, g


def run_TTA(sess, model, cfg, feed_dict):
    # Original
    probs = sess.run(model.probability, feed_dict=feed_dict)
    count = 1
    if cfg.eval_mirror and cfg.random_flip & 1 > 0:  # Left <-> Right
        new_feed_dict = {key: np.flip(im, axis=3) for key, im in feed_dict.items()}
        prob = sess.run(model.probability, feed_dict=new_feed_dict)
        probs += np.flip(prob, axis=3)
        count += 1
    if cfg.eval_mirror and cfg.random_flip & 2 > 0:  # Up <-> Down
        new_feed_dict = {key: np.flip(im, axis=2) for key, im in feed_dict.items()}
        prob = sess.run(model.probability, feed_dict=new_feed_dict)
        probs += np.flip(prob, axis=2)
        count += 1
    if cfg.eval_mirror and cfg.random_flip & 3 > 0:  # Left <-> Right Up <-> Down
        new_feed_dict = {key: np.flip(np.flip(im, axis=2), axis=3) for key, im in feed_dict.items()}
        prob = sess.run(model.probability, feed_dict=new_feed_dict)
        probs += np.flip(np.flip(prob, axis=2), axis=3)
        count += 1
    if cfg.eval_mirror and cfg.random_flip & 4 > 0:
        new_feed_dict = {key: np.flip(im, axis=1) for key, im in feed_dict.items()}
        prob = sess.run(model.probability, feed_dict=new_feed_dict)
        probs += np.flip(prob, axis=1)
        count += 1
    if cfg.eval_mirror and cfg.random_flip & 5 > 0:
        new_feed_dict = {key: np.flip(np.flip(im, axis=1), axis=3) for key, im in feed_dict.items()}
        prob = sess.run(model.probability, feed_dict=new_feed_dict)
        probs += np.flip(np.flip(prob, axis=1), axis=3)
        count += 1
    if cfg.eval_mirror and cfg.random_flip & 6 > 0:
        new_feed_dict = {key: np.flip(np.flip(im, axis=1), axis=2) for key, im in feed_dict.items()}
        prob = sess.run(model.probability, feed_dict=new_feed_dict)
        probs += np.flip(np.flip(prob, axis=1), axis=2)
        count += 1
    if cfg.eval_mirror and cfg.random_flip & 7 > 0:
        new_feed_dict = {key: np.flip(np.flip(np.flip(im, axis=1), axis=2), axis=3) for key, im in feed_dict.items()}
        prob = sess.run(model.probability, feed_dict=new_feed_dict)
        probs += np.flip(np.flip(np.flip(prob, axis=1), axis=2), axis=3)
        count += 1
    avg_prob = probs / count
    pred = np.argmax(avg_prob, axis=-1).astype(np.uint8)
    return pred


def main():
    cfg = _get_arguments()
    _custom_tf_logger(cfg)
    logging.debug(cfg)

    session_config = _get_session_config(cfg)
    d, h, w, c = (cfg.im_depth if cfg.im_depth > 0 else None,
                  cfg.im_height if cfg.im_height > 0 else None,
                  cfg.im_width if cfg.im_width > 0 else None,
                  cfg.im_channel)
    inputs, model, init, saver, graph = build_graph((1, d, h, w, c), cfg)

    # Initialize session
    sess = tf.Session(graph=graph, config=session_config)
    sess.run(init)
    checkpoint_path = find_checkpoint(cfg.ckpt_path, cfg.load_status_file, cfg)
    saver.restore(sess, checkpoint_path)

    dataset, nf_set, save_dir = load_dataset(cfg)
    accu = defaultdict(list)
    use_spatial = hasattr(cfg, "use_spatial") and cfg.use_spatial
    total_inters = [0, 0]
    for _, sample in nf_set.iterrows():
        # if sample.pid != 127:
        #     continue
        volume, label = dataset[sample.pid]["img"], dataset[sample.pid]["slim"]
        label_bool = label.astype(np.bool)
        if label.shape[0] % 2 != 0:
            label_bool = np.pad(label_bool, ((0, 1), (0, 0), (0, 0)), mode="constant", constant_values=0)

        # Run inference
        case_inters = [0, 0]
        img = volume.astype(np.float32)
        zoom_scale = np.array((1, cfg.im_height / img.shape[1], cfg.im_width / img.shape[2]), np.float32)
        img = ndi.zoom(img, zoom_scale, order=1)
        if img.shape[0] % 2 != 0:
            img = np.pad(img, ((0, 1), (0, 0), (0, 0)), mode="constant", constant_values=0)
        msk = img > 0
        tmp = img[msk]
        img[msk] = (tmp - tmp.mean()) / (tmp.std() + 1e-8)
        feed_dict = {inputs["images"]: img[None, ..., None]}
        pred = run_TTA(sess, model, cfg, feed_dict)
        final_pred = ndi.zoom(pred[0], 1 / zoom_scale, order=1)
        if label.shape[0] % 2 != 0:
            final_pred = final_pred[:-1]

        # Compute Metrics
        case_metrics = loss_metrics.metric_3d(final_pred, label, required=cfg.metrics_eval)
        conf = loss_metrics.ConfusionMatrix(final_pred, label)
        conf.compute()
        case_metrics.update({"fn": conf.fn, "fp": conf.fp, "tp": conf.tp})
        for key, value in case_metrics.items():
            accu[key].append(value)
        total_inters[0] += case_inters[0]
        total_inters[1] += case_inters[1]


if __name__ == "__main__":
    sys.exit(main())
