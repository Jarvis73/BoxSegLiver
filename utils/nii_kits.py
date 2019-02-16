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
import nibabel as nib   # conda install -c conda-forge nibabel

__all__ = [
    "nii_reader",
    "nii_writer"
]


def nii_reader(nii_path, only_meta=False):
    """ Implementation of `.nii` file reader

        Parameters
        ----------
        nii_path: str or Path
            file path to a nii file
        only_meta: bool
            if True, raw image will not be loaded and the second return is None

        Returns
        -------
        meta_info: dict
            a dictionary contains all the information in nii file
        raw_image: ndarray
            raw data of this image. If `only_meta` is True, this return will be None.

        Notes
        -----
        Order: (x, y, z) or (width, height, depth)
        Get spacing: (meta_info["srow_x"][0], meta_info["srow_y"][1], meta_info["srow_z"][2])
        Get shape: meta_info.get_data_shape()
        Get data type: meta_info["data_type"].dtype

        """
    data = nib.load(str(nii_path))
    meta_info = data.header

    raw_image = None
    if not only_meta:
        img = data.get_data()

        # get image orientation, a 4x4 matirx:
        #   flip up/down:       mat[0, 3] > 0
        #   flip left/right:    mat[1, 3] > 0
        qform = meta_info.get_qform()
        if qform[0, 3] < 0:
            img = np.flipud(img)
        if qform[1, 3] <= 0:
            img = np.fliplr(img)
        raw_image = np.transpose(img, (2, 1, 0)).astype(np.int16)

    return meta_info, raw_image


def nii_writer(nii_path, image, affine, header=None, verbose=False):
    new_image = nib.Nifti1Image(image, affine=affine, header=header)
    nib.save(new_image, str(nii_path))
    if verbose:
        print("Write nii to", nii_path)
