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
    # assert len(np.where(affine[:3, :3].reshape(-1) != 0)[0]) == 3, affine
    trans = np.argmax(np.abs(affine[:3, :3]), axis=1)
    data = nib_vol.get_fdata().astype(out_dtype).transpose(*trans[::-1])
    if special:
        data = np.flip(data, axis=2)
    if affine[0, trans[0]] > 0:                # Increase x from Right to Left
        data = np.flip(data, axis=2)
    if affine[1, trans[1]] > 0:                # Increase y from Anterior to Posterior
        data = np.flip(data, axis=1)
    if affine[2, trans[2]] < 0:                # Increase z from Interior to Superior
        data = np.flip(data, axis=0)
    return vh, data


def write_nii(data, header, out_path, out_dtype=np.int16, special=False, affine=None):
    if header is not None:
        affine = header.get_best_affine()
    assert len(np.where(affine[:3, :3].reshape(-1) != 0)[0]) == 3, affine
    trans = np.argmax(np.abs(affine[:3, :3]), axis=1)[::-1]
    trans_bk = [np.argwhere(np.array(trans) == i)[0][0] for i in range(3)]

    if special:
        data = np.flip(data, axis=2)
    if affine[0, trans[0]] > 0:  # Increase x from Right to Left
        data = np.flip(data, axis=2)
    if affine[1, trans[1]] > 0:  # Increase y from Anterior to Posterior
        data = np.flip(data, axis=1)
    if affine[2, trans[2]] < 0:  # Increase z from Interior to Superior
        data = np.flip(data, axis=0)

    out_image = np.transpose(data, trans_bk).astype(out_dtype)
    if header is None and affine is not None:
        out = nib.Nifti1Image(out_image, affine=affine)
    else:
        out = nib.Nifti1Image(out_image, affine=None, header=header)
    nib.save(out, str(out_path))
