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
from data_kits import analysis_kits
from utils import nii_kits

LiTS_ROOTS = [Path("D:\DataSet\LiTS\Training_Batch_1"),
              Path("D:\DataSet\LiTS\Training_Batch_2")]


def dump_all_liver_tumor_hist():
    save_dir = Path("D:\DataSet\LiTS\hist")
    for lits in LiTS_ROOTS:
        for image_path in lits.glob("volume-*.nii"):
            print(image_path)
            mask_path = image_path.parent / image_path.name.replace("volume", "segmentation")
            _, image = nii_kits.nii_reader(image_path)
            _, mask = nii_kits.nii_reader(mask_path)
            analysis_kits.compute_liver_tumor_hist(
                image, mask, 1, 2, image_path.stem,
                show=False, save_path=save_dir / image_path.with_suffix(".png").name)


def main():
    dump_all_liver_tumor_hist()


if __name__ == "__main__":
    main()
