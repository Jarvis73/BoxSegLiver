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

import json
import collections
import numpy as np
from pathlib import Path
import scipy.ndimage as ndi
import skimage.measure as measure
import nibabel as nib

from DataLoader.Liver import nii_kits
from interaction.liver_fw import Framework
from utils import array_kits


class SegViewerAdapter(object):
    def __init__(self, data_dir, lab_dir,
                 data_pat="test-volume-{}.nii",
                 lab_pat="test-segmentation-{}.nii",
                 meta_file=None):
        self.data_dir = Path(data_dir)
        self.lab_dir = Path(lab_dir)
        for dir_ in [self.data_dir, self.lab_dir]:
            assert dir_.exists(), str(dir_)
        self.data_pat = data_pat
        self.lab_pat = lab_pat
        self.meta_file = Path(meta_file)
        self.interaction = {}

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
        vol_file = self.data_dir / self.data_pat.format(self.pid)
        lab_file = self.lab_dir / self.lab_pat.format(self.pid)
        self.header, self.vol = nii_kits.read_nii(vol_file, np.float16)
        self.lab_bk = nii_kits.read_nii(lab_file, np.uint8)[1]
        # vv = nib.load(str(vol_file))
        # self.header = vv.header
        # self.vol = vv.get_fdata().astype(np.float16).transpose(2, 1, 0)
        # ll = nib.load(str(lab_file))
        # self.lab_bk = ll.get_fdata().astype(np.uint8).transpose(2, 1, 0)
        self.shape = self.vol.shape
        if self.meta_data is not None:
            self.bb = self.meta_data[self.pid]["bbox"]
            ranges = slice(self.get_min_idx(), self.get_max_idx())
            self.vol = self.vol[ranges]
            self.lab_bk = self.lab_bk[ranges]

        np.clip(self.vol, -200, 250, out=self.vol)
        np.add(self.vol, 200, out=self.vol)
        np.multiply(self.vol, 255 / 450, out=self.vol)
        self.vol = self.vol.astype(np.uint8)
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
        if sind in self.interaction[spid] and len(self.interaction[spid][sind]) > 0:
            ind = (ind - self.get_min_idx(ges)) % self.vol.shape[ges - 1]
            img = self.resized_image_single(self.vol, ges, ind)
            center = np.array([x["center"] for x in self.interaction[spid][sind]])
            stddev = np.array([x["stddev"] for x in self.interaction[spid][sind]])
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
                rng = self.meta_data[pid]["bbox"]
                self.table.append((path.name, "{}/{}".format(rng[3] - rng[0], size[0])))

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

    def update_interaction(self, ind, center, axes):
        sind = str(ind)
        spid = str(self.pid)
        center = [round(x, 3) for x in center]
        stddev = [round(x * 0.37065, 3) for x in axes]
        center = center[::-1]
        stddev = stddev[::-1]
        if sind in self.interaction[spid]:
            self.interaction[spid][sind].append({"center": list(center),
                                                 "stddev": stddev,
                                                 "z": [ind, ind + 1]})
        else:
            self.interaction[spid][sind] = [{"center": list(center), "stddev": stddev, "z": [ind, ind + 1]}]
        return {
            "PID": spid,
            "SID": sind,
            "z": "{}, {}".format(ind, ind + 1),
            "center": "{}, {}".format(*center),
            "stddev": "{}, {}".format(*stddev)
        }

    def pop_interaction(self, ind):
        sind = str(ind)
        spid = str(self.pid)
        if sind in self.interaction[spid]:
            self.interaction[spid][sind].pop()

    def save_interaction(self, save_path):
        res = {}
        for k, v in self.interaction.items():
            if v:
                case = {}
                for kk, vv in v.items():
                    if vv:
                        case[kk] = vv
                res[k] = case
        with Path(save_path).open("w") as f:
            json.dump(res, f)

    def load_interaction(self, load_path):
        with Path(load_path).open() as f:
            self.interaction = json.load(f)
        guide_list = []
        for pid, v in self.interaction.items():
            for sid, vv in v.items():
                for x in vv:
                    guide_list.append((pid, sid,
                                       "{}, {}".format(*x["z"]),
                                       "{}, {}".format(*x["center"]),
                                       "{}, {}".format(*x["stddev"])))
        return guide_list


def main():
    adapter = SegViewerAdapter(
        "E:/DataSet/LiTS/Training_Batch",
        # "E:/DataSet/LiTS/Test_Batch",
        "E:/DataSet/LiTS/Training_Batch",
        # Path(__file__).parent.parent / "model_dir/merge_001_unet_noise_0_05",
        # "E:/DataSet/LiTS/merge_013_gnet_sp_rand",
        # meta_file=Path(__file__).parent.parent / "DataLoader/Liver/prepare/test_meta_update.json",
        meta_file=Path(__file__).parent.parent / "DataLoader/Liver/prepare/meta.json",
        # data_pat="test-volume-{}.nii",
        # lab_pat="test-segmentation-{}.nii",
        data_pat="volume-{}.nii",
        lab_pat="segmentation-{}.nii",
    )

    demo = Framework(adapter, (512, 512))
    demo.configure_traits()


if __name__ == "__main__":
    main()
