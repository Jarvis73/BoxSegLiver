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
from skimage.morphology import skeletonize
import tensorflow as tf     # Tensorflow >= 1.13.0
from tensorflow.python.platform import tf_logging as logging

import config
import loss_metrics
from core import models
from DataLoader.NF import input_pipeline_g_simply as input_pipeline
from DataLoader import misc
from DataLoader.Liver import nii_kits
from utils.array_kits import create_gaussian_distribution_v2
from utils.ckpt_kits import find_checkpoint
from utils.logger import create_logger


def _get_arguments():
    parser = argparse.ArgumentParser()

    config.add_arguments(parser)
    models.add_arguments(parser)
    loss_metrics.add_arguments(parser)
    input_pipeline.add_arguments(parser)
    group = parser.add_argument_group(title="Evaluation Arguments")
    group.add_argument("--eval_final",
                       action="store_true",
                       required=False, help="Evaluate with final checkpoint. If not set, then evaluate "
                                            "with best checkpoint(default).")
    group.add_argument("--ckpt_path",
                       type=str,
                       required=False, help="Given a specified checkpoint for evaluation. "
                                            "(default best checkpoint)")
    group.add_argument("--save_path", type=str, default="prediction")
    group.add_argument("--inter_thresh", type=float, default=0.9, help="Threshold of the segmentation goal")
    group.add_argument("--max_iter", type=int, default=20, help="Maximum iters for each image in interaction")
    group.add_argument("--infer_set", type=str, default="eval", help="Choice from [train, eval]")
    group.add_argument("--save_subdir", type=str, default="prediction")

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
    data = input_pipeline.load_data()
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
    pred: np.ndarray, 2d, prediction
    ref:  np.ndarray, 2d, reference

    Returns
    -------
    pos: np.ndarray, a list of two values (y, x)
    fg: 0 for positive guide, 1 for negative
    """
    pred = pred.astype(np.bool)
    ref = ref.astype(np.bool)
    sym_diff = pred ^ ref
    struct = ndi.generate_binary_structure(2, 1)
    res, n_obj = ndi.label(sym_diff, struct)
    counts = np.bincount(res.flat)
    max_i = np.argmax(counts[1:]) + 1
    area = np.stack(np.where(res == max_i), axis=1)
    pos = np.mean(area, axis=0).round(0).astype(np.int32)
    if not sym_diff[pos[0], pos[1]]:
        ske = np.stack(np.where(skeletonize(sym_diff)), axis=1)
        min_i = np.argmin(np.sum((ske - pos) ** 2, axis=1))
        pos = ske[min_i]
    fg = 0 if ref[pos[0], pos[1]] else 1
    return pos, fg


def update_guide(pred, ref, guide, cfg, iteration):
    if pred is None:
        pred = np.zeros_like(ref, dtype=np.uint8)
    pos, fg = inter_simulation_test(pred, ref)
    # print(pos, pred.sum(), end=" ")
    cur_guide = create_gaussian_distribution_v2(
        ref.shape, [pos], [[cfg.stddev] * 2], euclidean=not cfg.local_enhance)
    if guide is None:
        if cfg.guide_channel == 2:
            guide = np.zeros(ref.shape + (2,), dtype=np.float32)
        else:
            guide = (np.zeros(ref.shape + (1,), dtype=np.float32),
                     np.zeros(ref.shape + (1,), dtype=np.float32))
    update_op = np.maximum if cfg.local_enhance else np.minimum
    if cfg.guide_channel == 2:
        guide[:, :, fg] = update_op(guide[:, :, fg], cur_guide) if guide[:, :, fg].max() > 0 else cur_guide
    else:
        guide[fg][:, :, 0] = update_op(guide[fg][:, :, 0], cur_guide)
    iteration[fg] += 1
    return guide, pos.tolist(), fg


def compute_dice(pred, ref):
    inter = np.count_nonzero(pred * ref)
    union = np.count_nonzero(pred) + np.count_nonzero(ref)
    return (2 * inter + 1e-8) / (union + 1e-8)


def _get_session_config(args):
    if args.device_mem_frac > 0:
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.device_mem_frac)
    else:
        gpu_options = tf.GPUOptions(allow_growth=True)
    sess_cfg = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)
    return sess_cfg


def build_graph(shape, cfg):
    bs, h, w, _ = shape
    with tf.Graph().as_default() as g:
        images = tf.placeholder(tf.float32, shape=shape, name="Image")
        inputs = {"images": images}

        if hasattr(cfg, "use_spatial") and cfg.use_spatial:
            sp_guide = tf.placeholder(tf.float32, shape=(bs, h, w, cfg.guide_channel), name="Guide")
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
    probs = []
    # Original
    prob = sess.run(model.probability, feed_dict=feed_dict)
    probs.append(prob)
    if cfg.eval_mirror and cfg.random_flip & 1 > 0:  # Left <-> Right
        new_feed_dict = {key: np.flip(im, axis=2) for key, im in feed_dict.items()}
        prob = sess.run(model.probability, feed_dict=new_feed_dict)
        probs.append(np.flip(prob, axis=2))
    if cfg.eval_mirror and cfg.random_flip & 2 > 0:  # Up <-> Down
        new_feed_dict = {key: np.flip(im, axis=1) for key, im in feed_dict.items()}
        prob = sess.run(model.probability, feed_dict=new_feed_dict)
        probs.append(np.flip(prob, axis=1))
    if cfg.eval_mirror and cfg.random_flip & 3 > 0:  # Left <-> Right Up <-> Down
        new_feed_dict = {key: np.flip(np.flip(im, axis=1), axis=2) for key, im in feed_dict.items()}
        prob = sess.run(model.probability, feed_dict=new_feed_dict)
        probs.append(np.flip(np.flip(prob, axis=1), axis=2))
    avg_prob = sum(probs) / len(probs)
    pred = np.argmax(avg_prob, axis=-1).astype(np.uint8)
    return pred


def main():
    cfg = _get_arguments()
    _custom_tf_logger(cfg)
    logging.debug(cfg)

    session_config = _get_session_config(cfg)
    h, w, c = (cfg.im_height if cfg.im_height > 0 else None,
               cfg.im_width if cfg.im_width > 0 else None,
               cfg.im_channel)
    inputs, model, init, saver, graph = build_graph((1, h, w, c), cfg)

    # Initialize session
    sess = tf.Session(graph=graph, config=session_config)
    sess.run(init)
    checkpoint_path = find_checkpoint(cfg.ckpt_path, cfg.load_status_file, cfg)
    saver.restore(sess, checkpoint_path)

    dataset, nf_set, save_dir = load_dataset(cfg)
    accu = defaultdict(list)
    use_spatial = hasattr(cfg, "use_spatial") and cfg.use_spatial
    shape = (cfg.im_width, cfg.im_height)
    total_inters = [0, 0]
    for _, sample in nf_set.iterrows():
        # if sample.pid != 127:
        #     continue
        volume, label = dataset[sample.pid]["img"], dataset[sample.pid]["slim"]
        _, height, width = volume.shape
        final_pred = np.zeros_like(label, dtype=np.uint8)

        # Run inference
        case_inters = [0, 0]
        slice_containing_nf = np.where(np.max(label, axis=(1, 2)) > 0)[0]
        for i in slice_containing_nf:
            # if i != 19:
            #     continue
            img = misc.img_crop(volume, i, cfg.im_channel)[0].transpose(1, 2, 0).astype(np.float32)
            img = cv2.resize(img, shape, interpolation=cv2.INTER_LINEAR)
            msk = img > 0
            tmp = img[msk]
            img[msk] = (tmp - tmp.mean()) / (tmp.std() + 1e-8)
            guide, pred, num_iter, pos = None, None, [0, 0], None
            if use_spatial:
                print("(Case {:3d} Slice {:2d}) Interacting: ".format(sample.pid, i), end="", flush=True)
                feed_dict = {inputs["images"]: img[None]}
                while True:
                    guide, new_pos, fg = update_guide(pred, label[i], guide, cfg, num_iter)
                    if new_pos == pos:
                        print(" Number iters: {}/{}".format(num_iter[0], num_iter[1]))
                        case_inters[0] += num_iter[0]
                        case_inters[1] += num_iter[1]
                        break
                    pos = new_pos
                    sp_guide = guide if cfg.guide_channel == 2 else guide[0] - guide[1]
                    sp_guide = cv2.resize(sp_guide, shape, interpolation=cv2.INTER_LINEAR)
                    sp_guide = sp_guide if cfg.local_enhance else \
                        (sp_guide / (256 * math.sqrt(2) * 0.8))
                    feed_dict[inputs["sp_guide"]] = sp_guide[None]
                    pred = run_TTA(sess, model, cfg, feed_dict)
                    pred = cv2.resize(pred[0], (width, height), interpolation=cv2.INTER_NEAREST)
                    dice = compute_dice(pred, label[i])
                    print("{:.3f} -> ".format(dice), end="", flush=True)
                    # with open("{}.pkl".format(num_iter), "wb") as f:
                    #     pickle.dump((img, sp_guide, pred), f)
                    if dice > cfg.inter_thresh or num_iter[0] + num_iter[1] >= cfg.max_iter:
                        print(" Number iters: {}/{}".format(num_iter[0], num_iter[1]))
                        case_inters[0] += num_iter[0]
                        case_inters[1] += num_iter[1]
                        break
                    # if num_iter > 4:
                    #     break
            else:
                feed_dict = {inputs["images"]: img[None]}
                pred = run_TTA(sess, model, cfg, feed_dict)
                pred = cv2.resize(pred[0], (width, height), interpolation=cv2.INTER_NEAREST)
            final_pred[i] = pred

        # Compute Metrics
        case_metrics = loss_metrics.metric_3d(final_pred, label, required=cfg.metrics_eval)
        conf = loss_metrics.ConfusionMatrix(final_pred, label)
        conf.compute()
        case_metrics.update({"fn": conf.fn, "fp": conf.fp, "tp": conf.tp})
        for key, value in case_metrics.items():
            accu[key].append(value)
        total_inters[0] += case_inters[0]
        total_inters[1] += case_inters[1]

        # Save prediction
        if cfg.save_predict:
            save_file = save_dir / f"predict-{sample.pid}.nii.gz"
            nii_kits.write_nii(final_pred, dataset[sample.pid]["meta"], save_file)
        logging.info(
            f"Evaluate-{sample.pid}" + "".join([" {}: {:.3f}".format(k, v) for k, v in case_metrics.items()]) +
            f" (Num inters: {case_inters})" * cfg.use_spatial + " (Saved)" * cfg.save_predict + "\n")

    tp, fp, fn = sum(accu.pop("tp")), sum(accu.pop("fp")), sum(accu.pop("fn"))
    accu = {k: np.mean(v) for k, v in accu.items()}
    accu["G_Dice"] = 2 * tp / (2 * tp + fn + fp)
    test_num = len(nf_set.index)
    logging.info(
        f"---Infer {len(nf_set.index)} cases" +
        "".join([" - {}: {:.3f}".format(k, v) for k, v in accu.items()]) +
        " (Average inters: {:.1f}/{:.1f})".format(
            total_inters[0] / test_num, total_inters[1] / test_num) * cfg.use_spatial
    )


if __name__ == "__main__":
    sys.exit(main())
