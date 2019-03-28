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
import scipy.ndimage as ndi
import matplotlib.pyplot as plt

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


def check_tumor_hist(num=10, xrng=(-200, 250), bins=100, yrng=(0, 0.02),
                     show=True, save_path=None):
    name = "volume-{}.nii".format(num)
    image_path = misc.find_file(LiTS_ROOTS, name)
    print(image_path)
    mask_path = image_path.parent / image_path.name.replace("volume", "segmentation")
    _, image = nii_kits.nii_reader(image_path)
    _, mask = nii_kits.nii_reader(mask_path)

    analysis_kits.compute_liver_tumor_hist(
        image, mask, 1, 2, "{} - total".format(image_path.stem),
        xrng=xrng, bins=bins, yrng=yrng,
        show=show, save_path=Path(save_path or "") / "0-total.png")

    # For each tumor
    disc = ndi.generate_binary_structure(3, connectivity=1)
    labeled, num_obj = ndi.label(array_kits.merge_labels(mask, [0, 2]), disc)
    mask = np.clip(mask, 0, 1) + labeled

    for i in range(num_obj):
        z, y, x = np.mean(np.array(np.where(mask == i + 2)), axis=1).astype(np.int32)
        analysis_kits.compute_liver_tumor_hist(
            image, mask, 1, 2 + i, "{} - tumor {} xyz: ({}, {}, {})"
            .format(image_path.stem, i + 1, x, y, z),
            xrng=xrng, bins=bins, yrng=yrng,
            show=show, save_path=Path(save_path or "") / "{}-tumor.png".format(i + 1))


def dump_all_tumor_hist():
    save_dir = Path("D:\DataSet\LiTS\hist_per_tumor")
    if not save_dir.exists():
        save_dir.mkdir(parents=True, exist_ok=True)

    for i in range(131):
        check_tumor_hist(i, show=False, save_path=Path(save_dir) / "volume-{}".format(i))


def check_np_hist(num=10, xrng=(-200, 250), bins=100, yrng=(0, 0.02)):
    name = "volume-{}.nii".format(num)
    image_path = misc.find_file(LiTS_ROOTS, name)
    print(image_path)
    mask_path = image_path.parent / image_path.name.replace("volume", "segmentation")
    _, image = nii_kits.nii_reader(image_path)
    _, mask = nii_kits.nii_reader(mask_path)

    a, b = image[mask == 1].flat, image[mask == 2].flat
    plt.hist(a, bins=bins, range=xrng, alpha=0.8, density=True)
    plt.hist(b, bins=bins, range=xrng, alpha=0.8, density=True)
    val3, bin3 = np.histogram(a, bins=bins, range=xrng, density=True)
    val4, bin4 = np.histogram(b, bins=bins, range=xrng, density=True)

    def foo(x):
        x = np.asarray(x)
        return (x[1:] + x[:-1]) / 2
    plt.plot(foo(bin3), val3)
    plt.plot(foo(bin4), val4)
    plt.xlim(xrng)
    plt.ylim(yrng)
    plt.show()


def main():
    # dump_all_liver_tumor_hist()
    # dump_all_tumor_det_metrics()
    # check_tumor_hist()
    # dump_all_tumor_hist()
    check_np_hist()


if __name__ == "__main__":
    main()
