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


def bbox_from_mask(mask, mask_values, min_shape=None, padding=None):
    """
    Calculate bounding box from a mask image

    Support N-D array

    Parameters
    ----------
    mask: ndarray
        a mask array
    mask_values: int or list or ndarray
        object values to compute bounding box
    min_shape: list or ndarray
        minimum shape of the returned bounding box(if small than min_shape, then enlarge the bounding box)
    padding: int or list or ndarray
        extra padding size along each axis [z, y, x]. It can be a integer, then all the padding size will
        be the same. If `padding` is not None, `min_shape` will be ignored.

    Returns
    -------
    bbox: ndarray, (x1, y1, x2, y2) or (x1, y1, z1, x2, y2, z2).

    Notes
    -----
    Both points (x1, y1, ...) and (x2, y2, ...) belong to the rectangular object region, which means that
    if you want to compute region size or create Python `slice` instance, then don't miss `+1`. It must be
    x2 - x1 + 1, y2 - y1 + 1, ...

    """
    # assert mask.ndim in [2, 3], "Wrong dimension of `mask`"

    if min_shape is not None:
        assert len(min_shape) == mask.ndim, "Dimensions are mismatch between `mask` and `min_shape`"

    if isinstance(mask_values, int):
        mask_values = [mask_values]
    mask_values = np.array(mask_values).reshape(-1, 1)

    indices = []
    for d in reversed(range(mask.ndim)):
        axes = list(range(mask.ndim))
        axes.remove(d)
        maxes = np.max(mask, axis=tuple(axes))
        indices.append(np.where((maxes == mask_values).any(axis=0))[0])

    coords = []     # (x1, x2, y1, y2, ...)
    for d in range(mask.ndim):
        coords.extend([indices[d][0], indices[d][-1]])
    coords = np.array(coords)

    if padding is None:
        if min_shape is None:
            min_shape = [0] * mask.ndim
        min_shape = np.array(min_shape)
        pad = np.clip((min_shape - (coords[1::2] - coords[::2] + 1)) / 2, 0, 65535)
    else:
        pad = padding / 2
    bbox = np.concatenate((np.maximum(0, coords[::2] - np.floor(pad[::-1]).astype(np.int32)),
                           np.minimum(np.array(mask.shape)[::-1] - 1,
                                      coords[1::2] + np.ceil(pad[::-1]).astype(np.int32))))

    return bbox


def merge_labels(masks, merges):
    """
    Convert multi-class labels to specific classes

    Support n-dim image
    """
    t_masks = np.zeros_like(masks, dtype=np.int8)  # use int8 to save memory
    for i, m_list in enumerate(merges):
        if isinstance(m_list, int):
            t_masks[np.where(masks == m_list)] = i
        elif isinstance(m_list, list):
            for m_digit in m_list:
                t_masks[np.where(masks == m_digit)] = i
        else:
            raise ValueError("Only integer or list is accepted, but got %r(type %s) in merges[%d]" % (m_list, type(m_list), i))
    return t_masks


def extract_object(src_image, src_mask=None):
    """

    Parameters
    ----------
    src_image: ndarray
        object image to extract
    src_mask: ndarray
        a mask specifying an object

    Returns
    -------
    object_image: ndarray, object image
    bbox: (x1, y1, ..., x2, y2, ...), optional

    """
    if src_mask is None:
        src_mask = src_image
    assert np.unique(src_mask).shape[0] == 2, "Mask should only contain two value {0, 1}"

    ndim = mask.ndim
    bbox = bbox_from_mask(mask, 1, padding=25)
    slices = [slice(bbox[d], bbox[d + ndim] + 1) for d in reversed(range(ndim))]
    object_image = src_image[tuple(slices)]

    return object_image, bbox


def find_empty_slices(src_image, axis=0, empty_value=0):
    """
    Find empty slices(zeros) along the specified axis.

    Parameters
    ----------
    src_image: ndarray
        an N-D image, for example [depth, height, width, channel]
    axis: int
        default 0
    empty_value: scalar
        background value

    Returns
    -------
    A 1-D boolean array that indicates empty slices

    """
    ndim = src_image.ndim
    axes = list(range(ndim))
    axes.remove(axis)
    empty_slices = np.all(src_image == empty_value, axis=axes)
    return empty_slices


if __name__ == "__main__":
    pass
    # from utils.mhd_kits import mhd_reader, mhd_writer, MyList
    #
    # meta, image = mhd_reader(r"D:\DataSet\LiTS\temp_3D\origin\T128.mhd")
    # _, mask = mhd_reader(r"D:\DataSet\LiTS\temp_3D\mask\T128_m.mhd")
    # mask = merge_labels(mask, [0, [255, 510]])
    # res, _ = extract_object(image, mask)
    # meta["DimSize"] = MyList(res.shape[::-1])
    # mhd_writer("./T128.mhd", res, meta, verbose=True)
