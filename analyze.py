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

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def glcm_stat():
    glcm_train_path = Path(__file__).parent / "data/LiTS/feat/glcm/train"
    glcm_test_path = Path(__file__).parent / "data/LiTS/feat/glcm/eval"

    all_glcm = []
    for feat in glcm_train_path.glob("*.npy"):
        cur_glcm = np.load(str(feat))
        cur_glcm = cur_glcm[cur_glcm.max(axis=1) > 0]
        all_glcm.append(cur_glcm)

    all_glcm_test = []
    for feat in glcm_test_path.glob("*.npy"):
        cur_glcm = np.load(str(feat))
        cur_glcm = cur_glcm[cur_glcm.max(axis=1) > 0]
        all_glcm_test.append(cur_glcm)

    all_glcm = np.concatenate(all_glcm, axis=0)
    all_glcm_test = np.concatenate(all_glcm_test, axis=0)
    return all_glcm, all_glcm_test


if __name__ == "__main__":
    tr, te = glcm_stat()
    feat = ["contrast", "dissimilarity", "homogeneity", "energy", "entropy", "correlation",
            "cluster_shade", "cluster_prominence"]
    save_path = Path(__file__).parent / "images_glcm_95p"
    save_path.mkdir(parents=True, exist_ok=True)

    all_scales = []
    for i in range(tr.shape[1]):
        obj = tr[:, i]
        p1, p2 = np.percentile(obj, [2.5, 97.5])
        obj = obj[np.logical_and(obj > p1, obj < p2)]
        # plt.hist(obj, bins=50)
        # plt.savefig(str(save_path / "{}_{}".format(feat[i // 12], i % 12)))
        # plt.close()
        all_scales.append((obj.max() - obj.min()) / 100)
    print(np.array(all_scales).round(4))

    all_scales = []
    for i in range(te.shape[1]):
        obj = te[:, i]
        p1, p2 = np.percentile(obj, [2.5, 97.5])
        obj = obj[np.logical_and(obj > p1, obj < p2)]
        # plt.hist(obj, bins=50)
        # plt.savefig(str(save_path / "{}_{}".format(feat[i // 12], i % 12)))
        # plt.close()
        all_scales.append((obj.max() - obj.min()) / 100)
    print(np.array(all_scales).round(4))

