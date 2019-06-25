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

import pickle
import numpy as np
from pathlib import Path
import scipy.ndimage as ndi
import skimage.measure as measure

from utils import nii_kits
from utils import array_kits
from utils.misc import find_file
from visualization.View_Kits import ComparePrediction
from visualization.Tool_Kits import get_pred_score


class SegViewerAdapter(object):
    def __init__(self, pred_dir, data_dirs, bbox_file=None):
        self.pred_dir = Path(pred_dir)
        self.data_dirs = [Path(data_dir) for data_dir in data_dirs]
        for dir_ in self.data_dirs + [self.pred_dir]:
            assert dir_.exists(), str(dir_)
        self.bbox_file = Path(bbox_file)

        self.gt = None
        self.gt1 = None
        self.pred_ = None
        self.pred = None
        self.mask_ = None
        self.mask = None
        self.liver = True
        self.liver_range = None
        self.bb = None
        self.label = 2
        self.table = []
        self.meta = None

    def update_case(self, case_path, **kwargs):
        self.gt = self.mask = self.mask_ = None
        self.pred = self.pred_ = None
        case_path = "prediction-{}.nii.gz".format(int(case_path.split("-")[1]))

        if Path(case_path).name.endswith(".nii.gz"):
            ori_file = find_file(self.data_dirs, case_path.replace("prediction", "volume")
                                 .replace(".nii.gz", ".nii"))
            lab_file = find_file(self.data_dirs, case_path.replace("prediction", "segmentation")
                                 .replace(".nii.gz", ".nii"))
            pred_file = find_file([self.pred_dir], case_path)
            reader = nii_kits.nii_reader
        else:
            raise ValueError("Wrong prediction name: {}".format(case_path))

        if self.liver_range is None and self.bbox_file.exists():
            with self.bbox_file.open("rb") as f:
                self.liver_range = pickle.load(f)

        self.meta, self.gt = reader(ori_file)
        self.pred_ = reader(pred_file)[1].astype(np.int8)
        self.mask_ = reader(lab_file)[1].astype(np.int8)
        self.shape = self.gt.shape
        if self.liver_range is not None:
            self.bb = self.liver_range[ori_file.name.split(".")[0]][0]
            ranges = slice(self.get_min_idx(), self.get_max_idx() + 1)
            self.gt = self.gt[ranges]
            self.pred_ = self.pred_[ranges]
            self.mask_ = self.mask_[ranges]

        np.clip(self.gt, -100, 400, self.gt)
        self.gt = ((self.gt + 100) * (255 / 500)).astype(np.uint8)

        self.liver = kwargs.get("liver", True)
        self.label = kwargs.get("label", 2)
        if self.liver:
            self.mask = array_kits.merge_labels(self.mask_, [0, [1, 2]]).astype(np.int8) * 2
            self.pred = array_kits.merge_labels(self.pred_, [0, [1, 2]]).astype(np.int8) * 2
        else:
            self.mask = array_kits.merge_labels(self.mask_, [0, 2]).astype(np.int8) * 2
            self.pred = array_kits.merge_labels(self.pred_, [0, self.label]).astype(np.int8) * 2

        assert self.gt.shape == self.pred.shape and self.gt.shape == self.mask.shape, \
            "gt: {}, mask: {}, pred: {}".format(self.gt.shape, self.mask.shape, self.pred.shape)

    def get_num_slices(self, ges=1):
        if self.gt is None:
            return 0
        return self.shape[ges - 1]

    def get_min_idx(self, ges=1):
        return max(self.bb[3 - ges] - 2, 0)

    def get_max_idx(self, ges=1):
        return min(self.bb[6 - ges] + 2, self.shape[0] - 1)

    def real_ind(self, ind, ges=1):
        if self.gt is None:
            return ind
        return (ind - self.get_min_idx(ges)) % self.gt.shape[ges - 1] + self.get_min_idx(ges)

    @staticmethod
    def plot_label(image, mask, color, contour, mask_lab, alpha):
        new_image = np.repeat(image[:, :, np.newaxis], 3, axis=2)
        if contour:
            contours = measure.find_contours(mask, 1)
            for cont in contours:
                cont = cont.astype(np.int16)
                new_image[cont[:, 0], cont[:, 1]] = np.array(color)
        elif mask_lab:
            masked = np.where(mask > 1)
            new_image[masked[0], masked[1]] = \
                (1 - alpha) * new_image[masked[0], masked[1]] + \
                alpha * np.array(color)

        return new_image

    @staticmethod
    def indices(ges, ind):
        slices = [slice(None)] * 3
        slices[ges - 1] = ind
        return slices

    def resized_image(self, im1, im2, ges, ind):
        spacing = [abs(self.meta["srow_z"][2]), abs(self.meta["srow_y"][1]), abs(self.meta["srow_x"][0])]
        if ges == 1:
            image, mask = im1[ind], im2[ind]
            return image, mask
        elif ges == 2:
            a, b = spacing[2], spacing[0]
            image, mask = im1[:, ind], im2[:, ind]
        else:
            a, b = spacing[2], spacing[1]
            image, mask = im1[:, :, ind], im2[:, :, ind]
        scale = [1, b / a] if b > a else [a / b, 1]
        image = ndi.zoom(image, scale, order=1)
        mask = ndi.zoom(mask, scale, order=0)
        return image, mask

    def get_slice1(self, ind, color=(255, 255, 255), alpha=0.3, **kwargs):
        ges = kwargs.get("ges", 1)
        ind = (ind - self.get_min_idx(ges)) % self.gt.shape[ges - 1]
        return self.plot_label(*self.resized_image(self.gt, self.mask, ges, ind),
                               color,
                               kwargs.get("contour", True),
                               kwargs.get("mask_lab", False),
                               alpha)

    def get_slice2(self, ind, color=(255, 255, 255), alpha=0.3, **kwargs):
        ges = kwargs.get("ges", 1)
        ind = (ind - self.get_min_idx(ges)) % self.gt.shape[ges - 1]
        contour = kwargs.get("contour", True)
        mask_lab = kwargs.get("mask_lab", False)

        return self.plot_label(*self.resized_image(self.gt, self.pred, ges, ind),
                               color, contour, mask_lab, alpha)

    def get_file_list(self):
        if self.liver_range is None and self.bbox_file.exists():
            with self.bbox_file.open("rb") as f:
                self.liver_range = pickle.load(f)

        if not self.table:
            for path in self.pred_dir.glob("*.nii.gz"):
                hdr, _ = nii_kits.nii_reader(str(path), only_meta=True)
                names = path.stem.split(".")[0].split("-")
                name = "Pred-{:03d}".format(int(names[1]))
                if self.liver_range is not None:
                    rng = self.liver_range[path.name.replace("prediction", "volume").split(".")[0]][0]
                    self.table.append((name, "{}/{}".format(rng[5] - rng[2] + 1,
                                                            hdr.get_data_shape()[-1])))
                else:
                    self.table.append((name, "{}".format(hdr.get_data_shape()[-1])))

        return self.table

    def get_pair_list(self, score_file):
        pairs = dict(get_pred_score(score_file))

        new_table = []
        for name, slices in self.table:
            score = pairs.get("volume-{}".format(int(name.split("-")[1])), (0.0, 0.0))
            new_table.append((name, slices, *score))

        return new_table

    def get_root_path(self):
        return str(self.pred_dir)

    def update_choice(self, **kwargs):
        self.liver = kwargs.get("liver", self.liver)
        self.label = kwargs.get("label", self.label)
        self.mask = array_kits.merge_labels(self.mask_,
                                            [0, [1, 2]] if self.liver else [0, 2]).astype(np.int8) * 2
        self.pred = array_kits.merge_labels(self.pred_,
                                            [0, [1, 2]] if self.liver else [0, self.label]).astype(np.int8) * 2

    def update_root_path(self, new_path):
        self.pred_dir = Path(new_path)
        self.table = []


def main():
    adapter = SegViewerAdapter(
        Path(__file__).parent / "model_dir/016_osmn_in_noise/prediction",
        ["D:/DataSet/LiTS/Training_Batch"],
        Path("D:/DataSet/LiTS/liver_bbox_nii.pkl")
    )

    demo = ComparePrediction(adapter)
    demo.configure_traits()


if __name__ == "__main__":
    main()
