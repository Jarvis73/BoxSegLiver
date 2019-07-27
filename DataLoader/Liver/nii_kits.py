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
import nibabel as nib


def read_lits(num, obj, file_name, only_header=False):
    if obj == "vol":
        return read_nii(file_name, out_dtype=np.int16,
                        special=True if 28 <= int(num) < 48 else False,
                        only_header=only_header)
    if obj == "lab":
        return read_nii(file_name, out_dtype=np.uint8,
                        special=True if 28 <= int(num) < 52 else False,
                        only_header=only_header)


def read_nii(file_name, out_dtype=np.int16, special=False, only_header=False):
    nib_vol = nib.load(str(file_name))
    vh = nib_vol.header
    if only_header:
        return vh
    affine = vh.get_best_affine()
    data = nib_vol.get_fdata().astype(out_dtype).transpose(2, 1, 0)
    if special:
        data = np.flip(data, axis=2)
    if affine[0, 0] > 0:                # Increase x from Right to Left
        data = np.flip(data, axis=2)
    if affine[1, 1] > 0:                # Increase y from Anterior to Posterior
        data = np.flip(data, axis=1)
    if affine[2, 2] < 0:                # Increase z from Interior to Superior
        data = np.flip(data, axis=0)
    return vh, data


def write_nii(data, header, out_path, out_dtype=np.int16, special=False):
    affine = header.get_best_affine()

    if special:
        data = np.flip(data, axis=2)
    if affine[0, 0] > 0:                # Increase x from Right to Left
        data = np.flip(data, axis=2)
    if affine[1, 1] > 0:                # Increase y from Anterior to Posterior
        data = np.flip(data, axis=1)
    if affine[2, 2] < 0:                # Increase z from Interior to Superior
        data = np.flip(data, axis=0)

    out_image = np.transpose(data, (2, 1, 0)).astype(out_dtype)
    out = nib.Nifti1Image(out_image, affine=None, header=header)
    nib.save(out, str(out_path))
