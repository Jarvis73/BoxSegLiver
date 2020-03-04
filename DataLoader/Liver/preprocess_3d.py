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
from pathlib import Path
from multiprocessing.pool import Pool
from skimage.transform import resize

from DataLoader.Liver import nii_kits

ROOT_DIR = Path(__file__).parent.parent.parent
target_spacing = np.array([1., 0.76757812, 0.76757812])


def load_save(args):
    nii_file, save_dir = args
    pid = int(nii_file.stem.split("-")[-1])
    print(nii_file)
    save_file = save_dir / "{}.npy".format(pid)
    vh, volume = nii_kits.read_lits(pid, "vol", nii_file)
    lab_file = nii_file.parent / nii_file.name.replace("volume", "segmentation")
    lh, label = nii_kits.read_lits(pid, "lab", lab_file)
    volume = (np.clip(volume.astype(np.float32), -200, 250) + 200) / 450.
    volume = resize(volume, (volume.shape[0], 256, 256), order=3, anti_aliasing=False)
    label = np.clip(label, 0, 1).astype(np.float32)
    label = resize(label, (label.shape[0], 256, 256), order=0)
    data = np.stack((volume, label), axis=-1)
    np.save(str(save_file), data)


def preprocess_liver_3d(save_dir):
    save_dir.mkdir(parents=True, exist_ok=True)
    source_dir = ROOT_DIR / "data/LiTS/Training_Batch"
    nii_list = list(source_dir.glob("volume-*.nii"))
    all_args = [(nii_file, save_dir) for nii_file, save_dir in zip(nii_list, [save_dir] * len(nii_list))]
    p = Pool(8)
    p.map(load_save, all_args)
    p.close()
    p.join()


if __name__ == "__main__":
    save_dir = ROOT_DIR / "data/LiTS/liver3d"
    preprocess_liver_3d(save_dir)
