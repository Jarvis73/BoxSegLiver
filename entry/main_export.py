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

import sys
import argparse
import tensorflow as tf     # Tensorflow >= 1.13.0
from pathlib import Path

import config
from core import models
from utils.ckpt_kits import find_checkpoint
from utils.image_ops import create_spatial_guide_2d


def _get_arguments():
    parser = argparse.ArgumentParser()

    config.add_arguments(parser)
    models.add_arguments(parser)
    group = parser.add_argument_group(title="Export Arguments")
    group.add_argument("--im_height", type=int, default=256)
    group.add_argument("--im_width", type=int, default=256)
    group.add_argument("--im_channel", type=int, default=3)
    group.add_argument("--use_spatial", action="store_true")
    group.add_argument("--guide_channel", type=int, default=2, help="1 or 2")

    group.add_argument("--ckpt_path",
                       type=str,
                       required=False, help="Given a specified checkpoint for evaluation. "
                                            "(default best checkpoint)")
    group.add_argument("--save_path", type=str, default="export")
    group.add_argument("--version", type=int, default=0)

    args = parser.parse_args()
    config.check_args(args, parser)
    config.fill_default_args(args)

    return args


def _get_session_config():
    gpu_options = tf.GPUOptions(allow_growth=True)
    sess_cfg = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)
    return sess_cfg


def zscore(img):
    nonzero_region = img > 0
    flatten_img = tf.reshape(img, [-1])
    flatten_mask = tf.reshape(nonzero_region, [-1])
    mean, variance = tf.nn.moments(tf.boolean_mask(flatten_img, flatten_mask), axes=(0,))
    float_region = tf.cast(nonzero_region, img.dtype)
    img = (img - float_region * mean) / (float_region * tf.math.sqrt(variance) + 1e-8)
    return img


def build_model(cfg):
    with tf.Graph().as_default() as g:
        src_shape = tf.placeholder(tf.int32, (3,))
        dst_shape = tf.placeholder(tf.int32, (2,))
        image = tf.placeholder(tf.float32)
        x = tf.expand_dims(tf.reshape(image, src_shape), axis=0)
        x = tf.image.resize_bilinear(x, dst_shape, align_corners=True)
        x = zscore(x)
        x = tf.identity(x, name="image")
        inputs = {"images": x}

        if hasattr(cfg, "use_spatial") and cfg.use_spatial:
            fg_pts = tf.placeholder(tf.float32, shape=(None, 4), name="FG_Pts")
            bg_pts = tf.placeholder(tf.float32, shape=(None, 4), name="BG_Pts")
            fg_ctr, fg_std = tf.split(fg_pts, 2, axis=1)
            bg_ctr, bg_std = tf.split(bg_pts, 2, axis=1)
            src_shape_float = tf.cast(src_shape[:2], tf.float32)
            dst_shape_float = tf.cast(dst_shape, tf.float32)
            fg_ctr = fg_ctr / src_shape_float * dst_shape_float
            bg_ctr = bg_ctr / src_shape_float * dst_shape_float
            fg_guide = create_spatial_guide_2d(dst_shape, fg_ctr, fg_std)
            bg_guide = create_spatial_guide_2d(dst_shape, bg_ctr, bg_std)
            sp_guide = tf.expand_dims(tf.concat((fg_guide, bg_guide), axis=-1), axis=0)
            guide = tf.identity(sp_guide, name="guide")
            inputs["sp_guide"] = guide

        model_params = models.get_model_params(cfg)
        model = model_params["model"](cfg)
        kwargs = model_params.get("model_kwargs", {})
        kwargs["ret_prob"] = False   # For TTA
        kwargs["ret_pred"] = False
        model(inputs, cfg.mode, *model_params.get("model_args", ()), **kwargs)

        logits_tf = model._layers["logits"]
        predictions_tf = tf.argmax(logits_tf, axis=3)
        predictions_tf = tf.squeeze(tf.squeeze(tf.image.resize_nearest_neighbor(
            tf.expand_dims(predictions_tf, axis=-1), src_shape[:2], align_corners=True), axis=-1), axis=0)
        saver = tf.train.Saver()
        return g, saver, (image, src_shape, dst_shape, fg_pts, bg_pts, predictions_tf)


def export_model(cfg, graph, saver, args):
    image, src_shape, dst_shape, fg_pts, bg_pts, predictions_tf = args

    with tf.Session(graph=graph, config=_get_session_config()) as sess:
        ckpt = find_checkpoint(cfg.ckpt_path, cfg.load_status_file, cfg)
        saver.restore(sess, ckpt)
        print("Model", cfg.tag, "restored.")

        export_path = Path(cfg.model_dir) / cfg.save_path / str(cfg.version)
        print('Exporting trained model to', str(export_path))
        builder = tf.saved_model.builder.SavedModelBuilder(str(export_path))

        tensor_info_image = tf.saved_model.utils.build_tensor_info(image)
        tensor_info_src_shape = tf.saved_model.utils.build_tensor_info(src_shape)
        tensor_info_dst_shape = tf.saved_model.utils.build_tensor_info(dst_shape)
        tensor_info_out_pred = tf.saved_model.utils.build_tensor_info(predictions_tf)
        export_inputs = {"images": tensor_info_image,
                         "src_shape": tensor_info_src_shape,
                         "dst_shape": tensor_info_dst_shape}
        export_outputs = {"out_pred": tensor_info_out_pred}

        if hasattr(cfg, "use_spatial") and cfg.use_spatial:
            tensor_info_fg_pts = tf.saved_model.utils.build_tensor_info(fg_pts)
            tensor_info_bg_pts = tf.saved_model.utils.build_tensor_info(bg_pts)
            export_inputs["fg_pts"] = tensor_info_fg_pts
            export_inputs["bg_pts"] = tensor_info_bg_pts

        prediction_signature = (
            tf.saved_model.signature_def_utils.build_signature_def(
                inputs=export_inputs,
                outputs=export_outputs,
                method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))
        signature_def_map = {'serving_default': prediction_signature}

        builder.add_meta_graph_and_variables(
            sess, [tf.saved_model.tag_constants.SERVING],
            signature_def_map=signature_def_map)

        builder.save(as_text=True)
        print('Done exporting!')


def run_model(cfg, graph, saver, pred_t, feed_dict):
    with tf.Session(graph=graph, config=_get_session_config()) as sess:
        ckpt = find_checkpoint(cfg.ckpt_path, cfg.load_status_file, cfg)
        saver.restore(sess, ckpt)
        print("Model", cfg.tag, "restored.")

        pred = sess.run(pred_t, feed_dict)
        return pred


def main():
    cfg = _get_arguments()
    graph, saver, in_out = build_model(cfg)
    export_model(cfg, graph, saver, in_out)
    # import pickle
    # import numpy as np
    # with open("data.pkl", "rb") as f:
    #     img = pickle.load(f).transpose(1, 2, 0).astype(np.float32)
    #
    # feed_dict = {in_out[0]: img.reshape(-1),
    #              in_out[1]: img.shape,
    #              in_out[2]: (960, 320),
    #              in_out[3]: np.array([[315., 217.6547, 5., 5.]], np.float32),
    #              in_out[4]: np.array([], np.float32).reshape(0, 4)}
    # pred = run_model(cfg, graph, saver, in_out[5], feed_dict)
    # print(pred.shape, pred.max(), pred.min(), np.bincount(pred.flat))


if __name__ == "__main__":
    sys.exit(main())
