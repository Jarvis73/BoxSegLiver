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

from core import models
from DataLoader import misc
from utils import array_kits


def _get_session_config():
    gpu_options = tf.GPUOptions(allow_growth=True)
    sess_cfg = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)
    return sess_cfg


class InferenceWithGuide2D(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.channel = 3
        self.stddev = 5.
        self.random_flip = 3
        shape = (1, None, None, self.channel)
        self.inputs, self.model, self.saver, self.g = self.build_graph(shape, cfg)
        self.sess = tf.Session(graph=self.g, config=_get_session_config())
        self.saver.restore(self.sess, self.cfg.ckpt_2d)

    def build_graph(self, shape, cfg):
        _, h, w, c = shape
        with tf.Graph().as_default() as g:
            images = tf.placeholder(tf.float32, shape=(1, h, w, c), name="ImageInfer")
            sp_guide = tf.placeholder(tf.float32, shape=(1, h, w, 2), name="GuideInfer")

            inputs = {"images": images, "sp_guide": sp_guide}
            model_params = models.get_model_2d_params(cfg)
            model = model_params["model"](cfg)
            kwargs = model_params.get("model_kwargs", {})
            kwargs["ret_prob"] = True  # For TTA
            kwargs["ret_pred"] = False
            model(inputs, "infer", *model_params.get("model_args", ()), **kwargs)

            saver = tf.train.Saver()
        return inputs, model, saver, g

    def run_TTA(self, feed_dict, do_mirror=False):
        probs = []
        prob = self.sess.run(self.model.probability, feed_dict=feed_dict)   # Original
        probs.append(prob)
        if do_mirror and self.random_flip & 1 > 0:  # Left <-> Right
            new_feed_dict = {key: np.flip(im, axis=2) for key, im in feed_dict.items()}
            prob = self.sess.run(self.model.probability, feed_dict=new_feed_dict)
            probs.append(np.flip(prob, axis=2))
        if do_mirror and self.random_flip & 2 > 0:  # Up <-> Down
            new_feed_dict = {key: np.flip(im, axis=1) for key, im in feed_dict.items()}
            prob = self.sess.run(self.model.probability, feed_dict=new_feed_dict)
            probs.append(np.flip(prob, axis=1))
        if do_mirror and self.random_flip & 3 > 0:  # Left <-> Right Up <-> Down
            new_feed_dict = {key: np.flip(np.flip(im, axis=1), axis=2) for key, im in feed_dict.items()}
            prob = self.sess.run(self.model.probability, feed_dict=new_feed_dict)
            probs.append(np.flip(np.flip(prob, axis=1), axis=2))
        avg_prob = sum(probs) / len(probs)
        pred = np.argmax(avg_prob, axis=-1).astype(np.uint8)
        return pred

    def get_pred_2d(self, image_patch, fg_pts, bg_pts):
        """
        Parameters
        ----------
        image_patch: np.ndarray, [depth, height, width]
        fg_pts: np.ndarray, [m, 3]
        bg_pts: np.ndarray, [n, 3]

        Returns
        -------
        pred_patch: np.ndarray, [depth, height, width]
        """
        ori_shape = np.array(image_patch.shape)
        cur_shape = np.array([ori_shape[0], self.cfg.im_height, self.cfg.im_width])
        scale = cur_shape / ori_shape
        image_patch = ndi.zoom(image_patch, scale, order=1)
        fg_pts *= scale
        bg_pts *= scale
        pred_patch = np.zeros_like(image_patch, np.float32)
        all_z = np.unique(fg_pts[:, 0])
        for z in all_z:
            z = int(z)
            fg = fg_pts[fg_pts[:, 0] == z, 1:]
            bg = bg_pts[bg_pts[:, 0] == z, 1:]
            image = misc.img_crop(image_patch, z, self.channel)[0].transpose(1, 2, 0).astype(np.float32)
            msk = image > 0
            tmp = image[msk]
            image[msk] = (tmp - tmp.mean()) / (tmp.std() + 1e-8)

            guide_shape = image.shape[:-1]
            if fg.shape[0] > 0:
                fg_guide = array_kits.create_gaussian_distribution_v2(
                    guide_shape, fg, np.ones_like(fg, np.float32) * self.stddev, keepdims=True)
            else:
                fg_guide = np.zeros(guide_shape + (1,), np.float32)
            if bg.shape[0] > 0:
                bg_guide = array_kits.create_gaussian_distribution_v2(
                    guide_shape, bg, np.ones_like(bg, np.float32) * self.stddev, keepdims=True)
            else:
                bg_guide = np.zeros(guide_shape + (1,), np.float32)
            guide = np.concatenate((fg_guide, bg_guide), axis=-1)
            feed_dict = {self.inputs["images"]: image[None],
                         self.inputs["sp_guide"]: guide[None]}
            # import pickle
            # with open(f"data_{z}.pkl", "wb") as f:
            #     pickle.dump((image, guide), f)
            pred_patch[z] = self.run_TTA(feed_dict)[0]
        pred_patch = ndi.zoom(pred_patch, 1. / scale, order=0)
        return pred_patch, all_z
