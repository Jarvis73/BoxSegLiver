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

import matplotlib.pyplot as plt
from pathlib import Path


def compute_liver_tumor_hist(image, mask, liver_lab=1, tumor_lab=2, title="",
                             bins=50, xrng=(0, 200), alpha=0.8, density=True,
                             yrng=(0, 1),
                             show=True, save_path=None):
    if not show and save_path is None:
        raise ValueError("If not show, save_path must be provided.")
    liver_volume = image[mask == liver_lab]
    tumor_volume = image[mask == tumor_lab]
    val1, bin1 = plt.hist(liver_volume.flat, bins=bins, range=xrng, alpha=alpha, density=density)
    val2, bin2 = plt.hist(tumor_volume.flat, bins=bins, range=xrng, alpha=alpha, density=density)
    plt.ylim(yrng)
    plt.legend(["Liver HU intensity", "Tumor HU intensity"])
    plt.xlabel("Intensity in CT Sequence")
    plt.ylabel("Normalized bin count")
    plt.title(title)

    if show:
        plt.show()
    else:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(save_path))
    plt.close()

    return val1, bin1, val2, bin2
