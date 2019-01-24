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

"""
Test for image_ops.py
"""

import unittest
import tensorflow as tf
from pathlib import Path
import matplotlib
matplotlib.use("qt5Agg")
import matplotlib.pyplot as plt

from utils import image_ops
from utils import mhd_kits


class TestImageOps(unittest.TestCase):
    def setUp(self):
        data_path = Path(__file__).parent.parent / "data/LiTS/Samples"
        _, self.image = mhd_kits.mhd_reader(data_path / "origin/T001.mhd")
        _, self.label = mhd_kits.mhd_reader(data_path / "mask/T001_m.mhd")
        image_array = self.image[58, ..., None]
        self.image_input = tf.convert_to_tensor(image_array, dtype=tf.float32)
        label_array = self.label[58]
        self.label_input = tf.convert_to_tensor(label_array, dtype=tf.int32)
        self.sess = tf.Session()

    def test_adjust_window_width_level(self):
        output = image_ops.adjust_window_width_level(self.image_input, 450, 25)
        output_array = self.sess.run(output)
        plt.imshow(output_array[0, ..., 0], cmap="gray")
        plt.show()

    def test_random_zoom_in_dim3(self):
        output = image_ops.adjust_window_width_level(self.image_input, 450, 25)
        output_image, output_label = image_ops.random_zoom_in(output, self.label_input,
                                                              seed_scale=None, seed_shift=None)
        output_array = self.sess.run([output_image, output_label])
        plt.subplot(121)
        plt.imshow(output_array[0][..., 0], cmap="gray")
        plt.subplot(122)
        plt.imshow(output_array[1], cmap="gray")
        plt.show()

    def test_random_zoom_in_dim4(self):
        output = image_ops.adjust_window_width_level(self.image_input, 450, 25)
        output_image, output_label = image_ops.random_zoom_in(output[None, ...], self.label_input[None, ...],
                                                              seed_scale=None, seed_shift=None)
        output_array = self.sess.run([output_image, output_label])
        plt.subplot(121)
        plt.imshow(output_array[0][0, ..., 0], cmap="gray")
        plt.subplot(122)
        plt.imshow(output_array[1][0], cmap="gray")
        plt.show()
