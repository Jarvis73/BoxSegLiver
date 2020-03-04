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

import cv2
import json
import collections
import numpy as np
from pathlib import Path
import scipy.ndimage as ndi
import skimage.measure as measure

from DataLoader.Liver import nii_kits
from interaction.nf_fw import Framework
from utils import array_kits


class SegViewerAdapter(object):
    def __init__(self, data_dir, lab_dir, json_file,
                 data_pat="volume-{}.nii",
                 lab_pat="segmentation-{}.nii",
                 meta_file=None, base_size=None):
        self.data_dir = Path(data_dir)
        self.lab_dir = Path(lab_dir)
        for dir_ in [self.data_dir, self.lab_dir]:
            assert dir_.exists(), str(dir_)
        self.data_pat = data_pat
        self.lab_pat = lab_pat
        self.meta_file = Path(meta_file)
        self.json_file = Path(json_file)
        self.base_size = base_size

        if self.json_file.exists():
            try:
                with self.json_file.open() as f:
                    self.interaction = json.load(f)
            except json.decoder.JSONDecodeError:
                self.interaction = collections.defaultdict(dict)
        else:
            self.interaction = collections.defaultdict(dict)

        self.empty = None
        self.ges = None
        self.vol = None
        self.lab_bk = None
        self.lab = None
        self.meta_data = None
        self.bb = None
        self.label = 0
        self.table = []
        self.header = None
        self.pid = None

        if self.meta_data is None and self.meta_file.exists():
            with self.meta_file.open() as f:
                self.meta_data = json.load(f)
            self.meta_data = {a["PID"]: a for a in self.meta_data}
        else:
            raise ValueError("meta_file is needed! meta_file:", self.meta_file)

    def update_case(self, case_path):
        self.gt = self.mask = self.mask_ = None
        self.pid = int(case_path.split(".")[0].split("-")[-1])
        vol_file = self.data_dir / self.data_pat.replace("{}", "{:03d}").format(self.pid)
        lab_file = self.lab_dir / self.lab_pat.replace("{}", "{:03d}").format(self.pid)

        self.header, self.vol = nii_kits.read_nii(vol_file, np.float16)
        self.lab_bk = np.clip(nii_kits.read_nii(lab_file, np.uint8)[1], 0, 1)
        self.lab_bk = cv2.resize(self.lab_bk.transpose((1, 2, 0)), self.base_size[::-1], interpolation=cv2.INTER_NEAREST)
        self.lab_bk = self.lab_bk.transpose((2, 0, 1))
        self.shape = self.vol.shape
        self.bb = [0] * 3 + self.meta_data[self.pid]["size"]
        self.my = self.shape[1] / self.base_size[0]
        self.mx = self.shape[2] / self.base_size[1]

        np.clip(self.vol, 0, 800, out=self.vol)
        np.multiply(self.vol, 255 / 800, out=self.vol)
        self.vol = self.vol.astype(np.uint8)
        self.vol = cv2.resize(self.vol.transpose((1, 2, 0)), self.base_size[::-1], interpolation=cv2.INTER_LINEAR)
        self.vol = self.vol.transpose((2, 0, 1))
        self.update_lab(self.label)

        assert self.vol.shape == self.lab.shape, "vol: {}, lab: {}".format(self.vol.shape, self.lab.shape)

    def get_num_slices(self, ges=1):
        if self.vol is None:
            return 0
        return self.shape[ges - 1]

    def get_min_idx(self, ges=1):
        return max(self.bb[ges - 1] - 2, 0)

    def get_max_idx(self, ges=1):
        return min(self.bb[2 + ges] + 2, self.shape[0] - 1)

    def real_ind(self, ind, ges=1):
        if self.vol is None:
            return ind
        return (ind - self.get_min_idx(ges)) % self.vol.shape[ges - 1] + self.get_min_idx(ges)

    def plot_label(self, image, mask, color, contour, alpha):
        new_image = np.repeat(image[:, :, np.newaxis], 3, axis=2)
        if contour:
            contours = measure.find_contours(mask, 0.5)
            for cont in contours:
                cont = cont.astype(np.int16)
                new_image[cont[:, 0], cont[:, 1]] = np.array(color)
        elif self.label:
            masked = np.where(mask == 1)
            new_image[masked[0], masked[1]] = (1 - alpha) * new_image[masked[0], masked[1]] + alpha * np.array(color)

        return new_image

    @staticmethod
    def indices(ges, ind):
        slices = [slice(None)] * 3
        slices[ges - 1] = ind
        return slices

    def resized_image(self, im1, im2, ges, ind):
        spacing = self.meta_data[self.pid]["spacing"]
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

    def resized_image_single(self, im, ges, ind):
        spacing = self.meta_data[self.pid]["spacing"]
        print(ges, ind, im.shape)
        if ges == 1:
            image = im[ind]
            return image
        elif ges == 2:
            a, b = spacing[2], spacing[0]
            image = im[:, ind]
        else:
            a, b = spacing[2], spacing[1]
            image = im[:, :, ind]
        scale = [1, b / a] if b > a else [a / b, 1]
        image = ndi.zoom(image, scale, order=1)
        return image

    def get_slice1(self, ind, color=(255, 255, 255), alpha=0.3, **kwargs):
        ges = kwargs.get("ges", 1)
        ind = (ind - self.get_min_idx(ges)) % self.vol.shape[ges - 1]
        return self.plot_label(*self.resized_image(self.vol, self.lab, ges, ind),
                               color,
                               kwargs.get("contour", True),
                               alpha)

    def get_slice2(self, ind, ges):
        sind = str(ind)
        spid = str(self.pid)
        if spid not in self.interaction:
            self.interaction[spid] = {}
        if sind in self.interaction[spid] and len(self.interaction[spid][sind]["centers"]) > 0:
            ind = (ind - self.get_min_idx(ges)) % self.vol.shape[ges - 1]
            img = self.resized_image_single(self.vol, ges, ind)
            center = np.array(self.interaction[spid][sind]["centers"]) / [self.my, self.mx]
            stddev = np.array(self.interaction[spid][sind]["stddevs"]) / [self.my, self.mx]
            img = array_kits.create_gaussian_distribution_v2(img.shape, center, stddev)
            return np.repeat(img[:, :, np.newaxis], 3, axis=2)
        else:
            if self.ges != ges:
                self.ges = ges
                shape = list(self.vol.shape)
                shape.remove(shape[ges - 1])
                self.empty = np.zeros(shape)
            return self.empty

    def get_file_list(self):
        if self.meta_data is None and self.meta_file.exists():
            with self.meta_file.open() as f:
                self.meta_data = json.load(f)
            self.meta_data = {a["PID"]: a for a in self.meta_data}

        if not self.table:
            for path in self.data_dir.glob(self.data_pat.format("*")):
                pid = int(path.name.split(".")[0].split("-")[-1])
                size = self.meta_data[pid]["size"]
                self.table.append((path.name, "{}".format(size[0])))

        return self.table

    def get_root_path(self):
        return str(self.data_dir)

    def update_lab(self, label):
        self.label = label
        if self.label == 0:
            self.lab = np.zeros_like(self.lab_bk, dtype=np.uint8)
        elif self.label == 1:
            self.lab = np.clip(self.lab_bk, 0, self.label)
        else:
            self.lab = self.lab_bk == self.label

    def update_root_path(self, new_path):
        self.data_dir = Path(new_path)
        self.table = []

    def update_interaction(self, ind, center, stddev):
        sind = str(ind)
        spid = str(self.pid)
        center = [round(center[1] * self.my, 3), round(center[0] * self.mx, 3)]
        stddev = [round(stddev[1] * 0.37065 * self.my, 3), round(stddev[0] * 0.37065 * self.mx, 3)]
        if sind in self.interaction[spid]:
            self.interaction[spid][sind]["centers"].append(list(center))
            self.interaction[spid][sind]["stddevs"].append(stddev)
        else:
            self.interaction[spid][sind] = {"centers": [list(center)], "stddevs": [stddev]}

    def pop_interaction(self, ind):
        sind = str(ind)
        spid = str(self.pid)
        if sind in self.interaction[spid]:
            self.interaction[spid][sind]["centers"].pop()
            self.interaction[spid][sind]["stddevs"].pop()

    def save_interaction(self):
        with self.json_file.open("w") as f:
            json.dump(self.interaction, f)


def main():
    adapter = SegViewerAdapter(
        r"D:\0WorkSpace\MedicalImageSegmentation\data\NF\nii_NF",
        # r"D:\0WorkSpace\MedicalImageSegmentation\data\LiTS\Test_Batch",
        r"D:\0WorkSpace\MedicalImageSegmentation\data\NF\nii_NF",
        # Path(__file__).parent.parent / "model_dir/merge_001_unet_noise_0_05",
        r"E:\Temp\112_nf_sp_dp\inter-87-10.json",
        meta_file=Path(__file__).parent.parent / "DataLoader/NF/prepare/meta.json",
        # data_pat="test-volume-{}.nii",
        # lab_pat="test-segmentation-{}.nii",
        data_pat="volume-{}.nii.gz",
        lab_pat="segmentation-{}.nii.gz",
        base_size=(1180, 322)
    )

    demo = Framework(adapter, (1180, 322))
    demo.configure_traits()


if __name__ == "__main__":
    main()
