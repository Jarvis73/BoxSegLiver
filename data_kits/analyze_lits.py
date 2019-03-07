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

from data_kits import analysis_kits
from utils import nii_kits
from utils import misc
import loss_metrics as metric_kits
from utils import array_kits


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


def dump_all_tumor_det_metrics():
    pred_dir = Path(__file__).parent.parent / "model_dir/004_triplet/prediction"
    save_file = Path(__file__).parent.parent / "model_dir/004_triplet/det_met.txt"
    with save_file.open("w") as f:
        metrics = []
        for res_path in pred_dir.glob("prediction-*.nii.gz"):
            ref_path = misc.find_file(LiTS_ROOTS, res_path.stem.replace("prediction", "segmentation"))
            _, res = nii_kits.nii_reader(res_path)
            res = np.clip(res - 1, 0, 1)
            _, ref = nii_kits.nii_reader(ref_path)
            ref = np.clip(ref - 1, 0, 1)
            met = metric_kits.tumor_detection_metrics(res, ref, verbose=True, name=res_path.stem)
            metrics.append(met)
            f.write("{:s} tp: {:d}, fp: {:d}, pos: {:d}, precision: {:.3f}, recall: {:.3f}\n"
                    .format(res_path.stem, met["tp"], met["fp"], met["pos"], met["precision"], met["recall"]))

        tps = np.sum([met["tp"] for met in metrics])
        fps = np.sum([met["fp"] for met in metrics])
        pos = np.sum([met["pos"] for met in metrics])

        print("#" * 80)
        f.write("#" * 80 + "\n")
        info = ("TPs: {:3d} FPs: {:3d} Pos: {:3d} Precision: {:.3f} Recall: {:.3f}"
                .format(tps, fps, pos, tps / (tps + fps), tps / pos))
        print(info)
        f.write(info)


def main():
    # dump_all_liver_tumor_hist()
    dump_all_tumor_det_metrics()


if __name__ == "__main__":
    main()
