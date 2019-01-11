# Copyright 2019 Zhang Jianwei All Right Reserved.
#
# TODO: Choice a License
#
# =================================================================================

import nibabel as nib   # conda install -c conda-forge nibabel


__all__ = [
    "nii_reader",
    "nii_writer"
]


def nii_reader(nii_path, only_meta=False):
    """ Implementation of `.mhd` file reader

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
