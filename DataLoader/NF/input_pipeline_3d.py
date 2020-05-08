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

import tqdm
import math
import zlib
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.ndimage.morphology import binary_erosion
from scipy.ndimage import label as label_connected, generate_binary_structure, find_objects
import tensorflow as tf
import tensorflow_estimator as tfes
from tensorflow.python.platform import tf_logging as logging
from tensorflow.contrib import data as contrib_data
from tensorflow.python.util import deprecation
from skimage.segmentation import find_boundaries
from scipy.ndimage.morphology import distance_transform_edt

from utils import image_ops
from utils import distribution_utils
from DataLoader import misc
# noinspection PyUnresolvedReferences
from DataLoader import feature_ops
from DataLoader.Liver import nii_kits
from entry import infer_2d

deprecation._PRINT_DEPRECATION_WARNINGS = False
ModeKeys = tfes.estimator.ModeKeys
Dataset = tf.data.Dataset
PROJ_ROOT = Path(__file__).parent.parent.parent
RND_SCALE = (1.0, 1.25)

data_cache = None
neg_cache = None
infer2d = None


def add_arguments(parser):
    group = parser.add_argument_group(title="Input Pipeline Arguments")
    group.add_argument("--test_fold", type=int, default=2)
    group.add_argument("--im_depth", type=int, default=10)
    group.add_argument("--im_height", type=int, default=256)
    group.add_argument("--im_width", type=int, default=256)
    group.add_argument("--im_channel", type=int, default=1)
    group.add_argument("--zoom_scale", type=float, nargs=2, default=RND_SCALE)
    group.add_argument("--random_flip", type=int, default=1,
                       help="Random flip while training. 0 for no flip, 1 for flip left/right, "
                            "2 for up/down, 4 for slices.")
    group.add_argument("--eval_in_patches", action="store_true")
    group.add_argument("--eval_num_batches_per_epoch", type=int, default=100)
    group.add_argument("--eval_mirror", action="store_true")
    group.add_argument("--tumor_percent", type=float, default=0.5)

    group = parser.add_argument_group(title="UNet3D Arguments")
    group.add_argument("--use_spatial", action="store_true")
    group.add_argument("--local_enhance", action="store_true")
    group.add_argument("--eval_no_sp", action="store_true", help="No spatial guide in evaluation")
    group.add_argument("--stddev", type=float, nargs="+", default=[1, 3., 3.],
                       help="Average stddev for spatial guide")
    group.add_argument("--save_sp_guide", action="store_true", help="Save spatial guide")
    group.add_argument("--eval_no_p", action="store_true", help="Evaluate with no propagation")
    group.add_argument("--guide_channel", type=int, default=2, help="1 or 2")
    group.add_argument("--fp_sample", action="store_true", help="Negative clicks sampled from false positive areas")
    group.add_argument("--sample_neg", type=float, default=0., help="Sample negative patches containing fp pixels")
    group.add_argument("--use_cascade", action="store_true", help="Enable cascade UNet(add extra channel with"
                                                                  " one labeled slice).")
    group.add_argument("--cascade_binary", action="store_true")
    group.add_argument("--use_2d", action="store_true", help="Use 2d model to generate prior")
    group.add_argument("-ds", "--downsampling", action="store_true", help="Use downsampled data for experiments")

    group = parser.add_argument_group(title="Net2D Arguments")
    group.add_argument("--model_2d", type=str)
    group.add_argument("--model_2d_config", type=str)
    group.add_argument("--ckpt_2d", type=str)


def load_data(debug=False, sub_dir="nii_NF", img_pattern="volume*", cache="cache"):
    global data_cache

    if data_cache is not None:
        return data_cache

    data_dir = PROJ_ROOT / "data/NF" / sub_dir
    path_list = list(data_dir.glob(img_pattern))

    if debug:
        path_list = path_list[:10]
        print(f"Loading data ({len(path_list)} examples, for debug) ...")
    else:
        print(f"Loading data ({len(path_list)} examples) ...")

    cache_path = PROJ_ROOT / f"data/NF/{cache}.gz.pkl"
    if cache_path.exists():
        print(f"Loading data cache from {cache_path}")
        with cache_path.open("rb") as f:
            data = zlib.decompress(f.read())
            data_cache = pickle.loads(data)
        print("Finished!")
        return data_cache

    data_cache = {}
    for path in tqdm.tqdm(path_list):
        pid = (path.name.split(".")[0].split("-")[-1] if sub_dir == "nii_NF" else
               path.name.split("-")[0])
        header, volume = nii_kits.read_nii(path)
        la_path = path.parent / path.name.replace("volume", "segmentation").replace("img", "mask")
        _, label = nii_kits.read_nii(la_path)
        if sub_dir == "test_NF":
            volume = volume.transpose(1, 0, 2)
            label = label.transpose(1, 0, 2)
        assert volume.shape == label.shape
        data_cache[int(pid)] = {"im_path": path.absolute(), "la_path": la_path.absolute(),
                                "img": volume,
                                "lab": label.astype(np.uint8),
                                "pos": np.stack(np.where(label > 0), axis=1),
                                "meta": header, "lab_rng": np.unique(label)}
    with cache_path.open("wb") as f:
        print(f"Saving data cache to {cache_path}")
        cache_s = pickle.dumps(data_cache, pickle.HIGHEST_PROTOCOL)
        f.write(zlib.compress(cache_s))
    print("Finished!")
    return data_cache


def load_data_ds(debug=False):
    """ Load data downsampling (for accelerating) """
    global data_cache

    if data_cache is not None:
        return data_cache

    data_dir = PROJ_ROOT / "data/NF/nii_NF"
    path_list = list(data_dir.glob("volume*"))

    if debug:
        path_list = path_list[:10]
        print(f"Loading data ({len(path_list)} examples, for debug) ...")
    else:
        print(f"Loading data ({len(path_list)} examples) ...")

    cache_path = PROJ_ROOT / "data/NF/cache_ds.gz.pkl"
    if cache_path.exists():
        print(f"Loading data cache from {cache_path}")
        with cache_path.open("rb") as f:
            data = zlib.decompress(f.read())
            data_cache = pickle.loads(data)
        print("Finished!")
        return data_cache

    data_cache = {}
    for path in tqdm.tqdm(path_list):
        pid = path.name.split(".")[0].split("-")[-1]
        header, volume = nii_kits.read_nii(path)
        la_path = path.parent / path.name.replace("volume", "segmentation")
        _, label = nii_kits.read_nii(la_path)
        assert volume.shape == label.shape
        volume = volume[:, ::2, ::2]
        label = label[:, ::2, ::2]
        data_cache[int(pid)] = {"im_path": path.absolute(), "la_path": la_path.absolute(),
                                "img": volume,
                                "lab": label.astype(np.uint8),
                                "pos": np.stack(np.where(label > 0), axis=1),
                                "meta": header, "lab_rng": np.unique(label)}
    with cache_path.open("wb") as f:
        print(f"Saving data cache to {cache_path}")
        cache_s = pickle.dumps(data_cache, pickle.HIGHEST_PROTOCOL)
        f.write(zlib.compress(cache_s))
    print("Finished!")
    return data_cache


def load_neg(data, dim=3):
    """ Load negative """
    global neg_cache

    if neg_cache is not None:
        return neg_cache

    neg_path = PROJ_ROOT / f"data/NF/neg_{dim}d.gz.pkl"

    if neg_path.exists():
        print(f"Loading negative cache from {neg_path}")
        with neg_path.open("rb") as f:
            neg_cache = pickle.loads(zlib.decompress(f.read()))
        print("Finished!")
        return neg_cache

    pred_dir = PROJ_ROOT / "model_dir/102_gnet_v3_2/train1"
    pred_list = list(pred_dir.glob("predict-*.nii.gz"))
    neg_cache = {}
    if dim == 3:
        struct = generate_binary_structure(3, 1)
        for path in tqdm.tqdm(pred_list):
            pid = int(path.name.split(".")[0].split("-")[-1])
            _, predict = nii_kits.read_nii(path)
            label = data[pid]["lab"]
            res, n_obj = label_connected(predict, struct)
            slices = find_objects(res)
            for i, sli in enumerate(slices):
                cube = res[sli]
                if ((cube == i + 1) * (label[sli] != 0)).sum() or (cube == i + 1).sum() <= 5:
                    cube[cube == i + 1] = 0
            result = np.clip(res, 0, 1).astype(np.uint8)
            neg_cache[pid] = {"bin": result, "pos": np.stack(np.where(result > 0), axis=1)}
    else:   # dim = 2
        struct = generate_binary_structure(2, 1)
        for path in tqdm.tqdm(pred_list):
            pid = int(path.name.split(".")[0].split("-")[-1])
            _, predict_3d = nii_kits.read_nii(path)
            label_3d = data[pid]["lab"].copy()
            result = np.zeros_like(predict_3d, dtype=np.uint8)
            for s in np.where(predict_3d.max(axis=(1, 2)))[0]:
                predict = predict_3d[s]
                label = label_3d[s]
                res, n_obj = label_connected(predict, struct)
                slices = find_objects(res)
                for i, sli in enumerate(slices):
                    cube = res[sli]
                    if ((cube == i + 1) * (label[sli] != 0)).sum() or (cube == i + 1).sum() <= 5:
                        cube[cube == i + 1] = 0
                result[s] = np.clip(res, 0, 1)
            neg_cache[pid] = {"bin": result, "pos": np.stack(np.where(result > 0), axis=1)}

    with neg_path.open("wb") as f:
        print(f"Saving negative cache to {neg_path}")
        f.write(zlib.compress(pickle.dumps(neg_cache, pickle.HIGHEST_PROTOCOL)))
    print("Finished!")
    return neg_cache


def load_split(test_fold=0, mode="train", filename="split.csv"):
    fold_path = PROJ_ROOT / "DataLoader/NF/prepare" / filename
    folds = pd.read_csv(str(fold_path))
    val_split = folds.loc[folds.split == test_fold]
    if mode != "train":
        return val_split
    train_folds = list(range(5))
    train_folds.remove(test_fold)
    train_split = folds.loc[folds.split.isin(train_folds)]
    return train_split


def inter_simulation(mask, margin=5, step=10, N=5, bg=False, d=40, strategy=0, ret_type=np.float32,
                     neg_patch=None):
    """
    Interaction simulation, including positive points and negative points

    Parameters
    ----------
    mask:     np.ndarray, binary mask, foreground points sampled from label=1 and bg from label=0
    margin:   int, margin band width in which no points are sampled
    step:     int, minimal distance between multiple interactions
    N:        int, maximum number of interctions
    bg:       bool, True for border_value=1, False for border_value=0 in binary erosion
    d:        int, band width outside the object
    strategy: int, value in [0, 1, 3, 4],
                   0: random in whole bg
                   1: random in band
                   3: surround the object evenly
                   4: random in false positive regions in `neg_patch`
    ret_type:
    neg_patch: np.ndarray, a mask contains false positive regions.

    Returns
    -------
    fg_pts: np.ndarray, shape [m, 3], corrdinates of the positive points
    bg_pts: np.ndarray, shape [n, 3], corrdinates of the negative points
    """
    small = False
    first = True
    all_pts = []
    if neg_patch is not None and strategy == 4:
        G = neg_patch.copy()
    else:
        struct = np.zeros((3, 3, 3), dtype=np.bool)
        struct[1] = True
        G = binary_erosion(mask, struct, iterations=margin, border_value=bg)
        if bg and strategy != 0:
            G = G ^ binary_erosion(G, struct, iterations=d, border_value=bg)
        if not G.max():    # too small tumor
            G = mask.copy()
            small = True

    depth, height, width = mask.shape
    for n in range(np.random.randint(int(not bg), N)):
        ctr = np.stack(np.where(G), axis=1)
        if not small:
            if first or strategy in [0, 1, 4]:
                i = np.random.choice(ctr.shape[0])
            else:  # strategy == 3
                dist = ctr.reshape(-1, 1, 3) - np.asarray(all_pts).reshape(1, -1, 3)
                i = np.argmax(np.sum(dist ** 2, axis=-1).min(axis=1))
            ctr = ctr[i]   # center z, y, x
        else:
            ctr = ctr.mean(axis=0).round().astype(np.int32)
        first = False
        all_pts.append(ctr)
        y1 = max(ctr[-2] - step, 0)
        y2 = min(ctr[-2] + step + 1, height)
        x1 = max(ctr[-1] - step, 0)
        x2 = min(ctr[-1] + step + 1, width)
        rcy, rcx, rh, rw = ctr[-2] - y1, ctr[-1] - x1, y2 - y1, x2 - x1   # relative center y, x, relative height, width
        Y, X = np.meshgrid(np.arange(rh), np.arange(rw), indexing="ij", sparse=True)
        circle = (X - rcx) ** 2 + (Y - rcy) ** 2 > step ** 2
        G[ctr[0], y1:y2, x1:x2] *= circle   # Only operate on single slice
        if small or not G.max():  # Cannot add more points
            break

    return np.asarray(all_pts, dtype=ret_type).reshape(-1, 3)


def input_fn(mode, params):
    if "args" not in params:
        raise KeyError("params of input_fn need an \"args\" key")
    args = params["args"]

    global infer2d
    if args.use_2d and infer2d is None:
        infer2d = infer_2d.InferenceWithGuide2D(args)
        logging.info(f"===== 2d model loaded from {args.ckpt_2d}")

    dataset = load_split(args.test_fold, mode)

    with tf.variable_scope("InputPipeline"):
        if mode == ModeKeys.TRAIN:
            return get_train_loader(dataset, cfg=args)
        elif mode == "eval_online":
            return get_val_loader(dataset, cfg=args)

#####################################
#
#   UNet input pipeline
#
#####################################


def data_processing(img, lab, *pts, train, cfg):
    # z_score
    nonzero_region = img > 0
    flatten_img = tf.reshape(img, [-1])
    flatten_mask = tf.reshape(nonzero_region, [-1])
    mean, variance = tf.nn.moments(tf.boolean_mask(flatten_img, flatten_mask), axes=(0,))
    float_region = tf.cast(nonzero_region, img.dtype)
    img = (img - float_region * mean) / (float_region * tf.math.sqrt(variance) + 1e-8)
    flag = 'Train' if train else 'Val'
    logging.info(f"{flag}: Add z-score preprocessing")
    logging.info(f"{flag}: {'Enable' if cfg.local_enhance else 'Disable'} local enhance of the guides")

    if cfg.use_spatial:
        fg_pts, bg_pts = pts

        def true_fn(ctr):
            stddev = tf.ones(tf.shape(ctr), tf.float32) * cfg.stddev
            gd = image_ops.create_spatial_guide_3d(
                tf.shape(img)[:-1], ctr, stddev, euclidean=not cfg.local_enhance)
            if not cfg.local_enhance:
                gd = gd / (cfg.im_height * math.sqrt(2) * 0.8)  # normalization
            return tf.cast(gd, tf.float32)

        def false_fn():
            return tf.zeros(tf.concat([tf.shape(img)[:-1], [1]], axis=0), tf.float32)

        fg_guide = tf.cond(tf.shape(fg_pts)[0] > 0, lambda: true_fn(fg_pts), false_fn)
        bg_guide = tf.cond(tf.shape(bg_pts)[0] > 0, lambda: true_fn(bg_pts), false_fn)
        img = tf.concat([img, fg_guide, bg_guide], axis=-1)
    img = tf.image.resize_bilinear(img, (cfg.im_height, cfg.im_width), align_corners=True)

    if train and cfg.random_flip > 0:
        img, lab = image_ops.random_flip(img, lab, flip=cfg.random_flip)
        logging.info("Train: Add random flip " +
                     "left <-> right " * (cfg.random_flip & 1 > 0) +
                     "up <-> down" * (cfg.random_flip & 2 > 0) +
                     "front <-> back" * (cfg.random_flip & 4 > 0))

    if cfg.use_spatial:
        if cfg.guide_channel == 2:
            img, sp_guide = tf.split(img, [cfg.im_channel, 2], axis=-1)
        else:
            img, fg_guide, bg_guide = tf.split(img, [cfg.im_channel, 1, 1], axis=-1)
            sp_guide = fg_guide - bg_guide
        feat = {"images": img, "sp_guide": sp_guide}
    else:
        feat = {"images": img}

    lab = tf.expand_dims(lab, axis=-1)
    lab = tf.image.resize_nearest_neighbor(lab, (cfg.im_height, cfg.im_width), align_corners=True)
    lab = tf.squeeze(lab, axis=-1)

    if train:
        feat["images"] = image_ops.augment_gamma(feat["images"], gamma_range=(0.7, 1.5), retain_stats=True,
                                                 p_per_sample=0.3)
        logging.info("Train: Add gamma transform, scale=(0.7, 1.5), retain_stats=True, p_per_sample=0.3")
    return feat, lab


def data_processing_2c(img, lab, *pts, train, cfg):
    """ img has two channels: [img, res2d] """
    img, res2d = tf.split(img, 2, axis=-1)
    # z_score
    nonzero_region = img > 0
    flatten_img = tf.reshape(img, [-1])
    flatten_mask = tf.reshape(nonzero_region, [-1])
    mean, variance = tf.nn.moments(tf.boolean_mask(flatten_img, flatten_mask), axes=(0,))
    float_region = tf.cast(nonzero_region, img.dtype)
    img = (img - float_region * mean) / (float_region * tf.math.sqrt(variance) + 1e-8)
    img = tf.concat((img, res2d), axis=-1)
    flag = 'Train' if train else 'Val'
    logging.info(f"{flag}: Add z-score preprocessing")
    logging.info(f"{flag}: {'Enable' if cfg.local_enhance else 'Disable'} local enhance of the guides")

    if cfg.use_spatial:
        fg_pts, bg_pts = pts

        def true_fn(ctr):
            stddev = tf.ones(tf.shape(ctr), tf.float32) * cfg.stddev
            gd = image_ops.create_spatial_guide_3d(
                tf.shape(img)[:-1], ctr, stddev, euclidean=not cfg.local_enhance)
            if not cfg.local_enhance:
                gd = gd / (cfg.im_height * math.sqrt(2) * 0.8)  # normalization
            return tf.cast(gd, tf.float32)

        def false_fn():
            return tf.zeros(tf.concat([tf.shape(img)[:-1], [1]], axis=0), tf.float32)

        fg_guide = tf.cond(tf.shape(fg_pts)[0] > 0, lambda: true_fn(fg_pts), false_fn)
        bg_guide = tf.cond(tf.shape(bg_pts)[0] > 0, lambda: true_fn(bg_pts), false_fn)
        img = tf.concat([img, fg_guide, bg_guide], axis=-1)
    img = tf.image.resize_bilinear(img, (cfg.im_height, cfg.im_width), align_corners=True)

    if train and cfg.random_flip > 0:
        img, lab = image_ops.random_flip(img, lab, flip=cfg.random_flip)
        logging.info("Train: Add random flip " +
                     "left <-> right " * (cfg.random_flip & 1 > 0) +
                     "up <-> down" * (cfg.random_flip & 2 > 0) +
                     "front <-> back" * (cfg.random_flip & 4 > 0))

    if cfg.use_spatial:
        if cfg.guide_channel == 2:
            img, res2d, sp_guide = tf.split(img, [cfg.im_channel - 1, 1, 2], axis=-1)
        else:
            img, res2d, fg_guide, bg_guide = tf.split(img, [cfg.im_channel - 1, 1, 1, 1], axis=-1)
            sp_guide = fg_guide - bg_guide
        feat = {"images": img, "sp_guide": sp_guide}
    else:
        feat = {"images": img}

    lab = tf.expand_dims(lab, axis=-1)
    lab = tf.image.resize_nearest_neighbor(lab, (cfg.im_height, cfg.im_width), align_corners=True)
    lab = tf.squeeze(lab, axis=-1)

    if train:
        feat["images"] = image_ops.augment_gamma(feat["images"], gamma_range=(0.7, 1.5), retain_stats=True,
                                                 p_per_sample=0.3)
        logging.info("Train: Add gamma transform, scale=(0.7, 1.5), retain_stats=True, p_per_sample=0.3")
    feat["images"] = tf.concat((feat["images"], res2d), axis=-1)
    return feat, lab


def gen_kernel(nf, img_patch, lab_patch, cfg):
    """
    Kernel for parallel processing

    Parameters
    ----------
    nf:         bool, has NF or not
    img_patch:  np.ndarray, image patch
    lab_patch:  np.ndarray, label patch
    cfg:
    """
    try:
        if cfg.fp_sample:
            neg_patch = lab_patch // 10
            lab_patch = lab_patch % 10
        else:
            neg_patch = None
        if lab_patch.max() > 0:     # Sample positive clicks
            fg_pts = inter_simulation(lab_patch, margin=3, step=10, N=5, bg=False, strategy=0) \
                if nf else np.zeros((0, 2), dtype=np.float32)
        else:   # No objects
            fg_pts = np.zeros((0, 3), dtype=np.float32)

        if neg_patch is not None and neg_patch.max() > 0:
            strategy = 4
        elif np.random.sample() > 0.5:
            strategy = 1
        else:
            strategy = 3
        bg_pts = inter_simulation(1 - lab_patch, margin=3, step=10, N=5, bg=True, d=40,
                                  strategy=strategy, neg_patch=neg_patch)
        if cfg.use_cascade:
            if fg_pts.size == 0:
                exp_dist = np.zeros_like(img_patch, np.float32)
            else:
                if cfg.use_2d:
                    prior, all_z = infer2d.get_pred_2d(img_patch[..., 0], fg_pts, bg_pts)
                    # prior2 = prior.copy()
                    if not cfg.cascade_binary:
                        for z in all_z:
                            z = int(z)
                            prior[z] = find_boundaries(prior[z], mode='inner')
                        dist = distance_transform_edt(np.bitwise_not(prior.astype(np.bool)))
                        exp_dist = np.exp(-dist / 25)[..., None]
                    else:
                        exp_dist = prior[..., None]
                else:
                    z = int(fg_pts[0, 0])
                    prior = lab_patch[z]
                    if not cfg.cascade_binary:
                        boundaries = find_boundaries(prior, mode='inner')
                        lab_patch_from_2d = np.zeros_like(lab_patch, np.bool)
                        lab_patch_from_2d[z] = boundaries
                        dist = distance_transform_edt(np.bitwise_not(lab_patch_from_2d))
                        exp_dist = np.exp(-dist / 25)[..., None]
                    else:
                        lab_patch_from_2d = np.zeros_like(lab_patch, np.float32)
                        lab_patch_from_2d[z] = prior
                        exp_dist = lab_patch_from_2d
            img_patch = np.concatenate((img_patch.astype(np.float32), exp_dist), axis=-1)
            # with open("data.pkl", "wb") as f:
            #     pickle.dump((img_patch, prior2, fg_pts, bg_pts), f)
            #     exit()
        else:
            img_patch = img_patch.astype(np.float32)
        return img_patch, lab_patch.astype(np.int32), fg_pts, bg_pts
    except (KeyboardInterrupt, FileNotFoundError, EOFError):
        logging.info("Subprocess terminated.")


def gen_batch(dataset, batch_size, train, cfg):
    """
    Batch sampler (return a generator)

    Parameters
    ----------
    dataset: pandas.DataFrame, two column [split, pid]
    batch_size: int, batch size for each replica
    train: bool, train/val
    cfg: global config var

    """
    data = load_data() if not cfg.downsampling else load_data_ds()
    for k, v in data.items():
        np.clip(v["lab"], 0, 1, v["lab"])
    if cfg.sample_neg or cfg.fp_sample:
        neg_data = load_neg(data)
    dataset = dataset[[True if pid in data else False for pid in dataset.pid]]
    # dataset containing nf (remove benign scans)
    dataset['nf'] = [True if len(data[pid]['lab_rng']) > 1 else False for pid in dataset.pid]
    nf_set = dataset[dataset.nf]
    force_tumor = math.ceil(batch_size * cfg.tumor_percent)
    force_fp = math.ceil(batch_size * cfg.sample_neg)

    target_size = np.array([cfg.im_depth, cfg.im_height, cfg.im_width], dtype=np.float32)
    zoom = cfg.zoom_scale
    if not train:
        np.random.seed(1234)    # fix validation batches
        zoom = (1.125, 1.125)

    while True:
        nf = nf_set.sample(n=force_tumor, replace=False, weights=None)
        nf['flag'] = [1] * len(nf.index)
        if cfg.sample_neg:
            fp = nf_set.sample(n=force_fp, replace=False, weights=None)
            fp['flag'] = [2] * len(fp.index)
            batch = pd.concat([nf, fp])
        else:
            rem = dataset[~dataset.index.isin(nf.index)].sample(
                n=batch_size - force_tumor, replace=False, weights=None)
            rem['flag'] = [0] * len(rem.index)
            batch = pd.concat([nf, rem])

        # queue = []
        for i, sample in batch.iterrows():     # columns [split, pid, nf(bool)]
            crop_shape = (target_size * ([1] + np.random.uniform(*zoom, size=2).tolist())).astype(np.int32)
            depth, height, width = data[sample.pid]['img'].shape  # volume shape
            if sample.flag == 1:
                i = np.random.choice(data[sample.pid]['pos'].shape[0])  # choice a foreground pixel
                pz, py, px = data[sample.pid]['pos'][i]
            elif train and sample.flag == 2 and neg_data[sample.pid]['pos'].shape[0] > 0:
                i = np.random.choice(neg_data[sample.pid]['pos'].shape[0])
                pz, py, px = neg_data[sample.pid]['pos'][i]
            else:
                pz = np.random.randint(depth)
                py = np.random.randint(height)
                px = np.random.randint(width)
            img_patch, slices = misc.volume_crop(data[sample.pid]['img'], (pz, py, px), crop_shape)
            lab_patch = data[sample.pid]['lab'][slices]
            if train and cfg.fp_sample:
                neg_patch = neg_data[sample.pid]["bin"][slices]
                lab_patch = lab_patch + neg_patch * 10

            img_patch = img_patch[..., None]
            if cfg.use_spatial:
                lab_patch = lab_patch.astype(np.int8)
                yield gen_kernel(sample.nf, img_patch, lab_patch, cfg)
            else:
                img_patch = img_patch.astype(np.float32)
                lab_patch = lab_patch.astype(np.int32)
                yield img_patch, lab_patch


def get_train_loader(data_list, cfg):
    batch_size = distribution_utils.per_device_batch_size(cfg.batch_size, cfg.num_gpus)
    if cfg.zoom_scale[1] > cfg.zoom_scale[0]:
        logging.info("Train: Add random zoom, scale = ({}, {})".format(*cfg.zoom_scale))

    def train_gen():
        return gen_batch(data_list, batch_size, train=True, cfg=cfg)

    output_types = (tf.float32, tf.int32)
    output_shapes = (tf.TensorShape([None, None, None, cfg.im_channel]), tf.TensorShape([None, None, None]))
    if cfg.use_spatial:
        output_types = output_types + (tf.float32, tf.float32)
        output_shapes = output_shapes + (tf.TensorShape([None, 3]), tf.TensorShape([None, 3]))

    processing = data_processing if not cfg.use_cascade else data_processing_2c
    dataset = (tf.data.Dataset.from_generator(train_gen, output_types, output_shapes)
               .apply(tf.data.experimental.map_and_batch(
                        lambda *args: processing(*args, train=True, cfg=cfg),
                        batch_size, num_parallel_batches=1))
               .prefetch(buffer_size=contrib_data.AUTOTUNE))

    return dataset


def get_val_loader(data_list, cfg):
    batch_size = distribution_utils.per_device_batch_size(cfg.batch_size, cfg.num_gpus)

    def eval_gen():
        infinity_generator = gen_batch(data_list, batch_size, train=False, cfg=cfg)
        for _ in tqdm.tqdm(range(cfg.eval_num_batches_per_epoch * cfg.batch_size)):
            yield next(infinity_generator)

    output_types = (tf.float32, tf.int32)
    output_shapes = (tf.TensorShape([None, None, None, cfg.im_channel]), tf.TensorShape([None, None, None]))
    if cfg.use_spatial:
        output_types = output_types + (tf.float32, tf.float32)
        output_shapes = output_shapes + (tf.TensorShape([None, 3]), tf.TensorShape([None, 3]))

    processing = data_processing if not cfg.use_cascade else data_processing_2c
    dataset = (tf.data.Dataset.from_generator(eval_gen, output_types, output_shapes)
               .apply(tf.data.experimental.map_and_batch(
                        lambda *args: processing(*args, train=False, cfg=cfg),
                        batch_size, num_parallel_batches=1))
               .prefetch(buffer_size=contrib_data.AUTOTUNE))

    return dataset
