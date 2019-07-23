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
import matplotlib.pyplot as plt
from pathlib import Path


def visual_hist(num):
    path = Path(__file__).parent.parent.parent / "data/LiTS/feat/hist/train/{:03d}.npy".format(num)
    d1 = np.load(str(path), allow_pickle=True)
    path = Path(__file__).parent.parent.parent / "data/LiTS/feat/hist/eval/{:03d}.npy".format(num)
    d2 = np.load(str(path), allow_pickle=True)
    plt.subplot(221)
    plt.imshow(d1[:, :100])
    plt.colorbar()
    plt.subplot(223)
    plt.imshow(d1[:, 100:])
    plt.colorbar()
    plt.subplot(222)
    plt.imshow(d2[:, :100])
    plt.colorbar()
    plt.subplot(224)
    plt.imshow(d2[:, 100:])
    plt.colorbar()
    plt.show()


if __name__ == "__main__":
    visual_hist(3)
