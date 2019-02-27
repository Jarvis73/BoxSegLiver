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
Test for array_kits.py
"""

import unittest
from pathlib import Path
import matplotlib
matplotlib.use("qt5Agg")
import matplotlib.pyplot as plt

from utils import array_kits
from utils import mhd_kits


class TestImageOps(unittest.TestCase):
    def setUp(self):
        data_path = Path(__file__).parent.parent / "data/LiTS/Samples"
        _, self.image = mhd_kits.mhd_reader(data_path / "T001.mhd")
        _, self.label = mhd_kits.mhd_reader(data_path / "T001_m.mhd")
        self.image = array_kits.aug_window_width_level(self.image, 450, 50)
        self.label = array_kits.merge_labels(self.label, [0, 255, 510])

    def test_get_gd_image_multi_objs(self):
        for i in range(44, self.label.shape[0]):
            guide = array_kits.get_gd_image_multi_objs(self.label[i], 2, with_fake_guides=True)
            plt.subplot(131)
            plt.imshow(self.image[i], cmap="gray")
            plt.subplot(132)
            plt.imshow(self.label[i], cmap="gray")
            plt.subplot(133)
            plt.imshow(guide, cmap="gray")
            plt.show()
