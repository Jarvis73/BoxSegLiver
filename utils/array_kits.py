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

import random
import functools
import numpy as np
import scipy.ndimage as ndi


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
    if np.count_nonzero(mask) == 0:
        print("Warning: mask is empty!")
        return np.zeros(shape=(mask.ndim * 2,))

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
        elif isinstance(m_list, (list, tuple)):
            for m_digit in m_list:
                t_masks[np.where(masks == m_digit)] = i
        else:
            raise ValueError("Only integer or list is accepted, but got %r(type %s) in merges[%d]" % (m_list, type(m_list), i))
    return t_masks


def bbox_to_slices(bbox):
    if not isinstance(bbox, (list, tuple, np.ndarray)):
        raise TypeError("`bbox` must be an iterable object, got {}"
                        .format(type(bbox)))
    if isinstance(bbox, np.ndarray) and len(bbox.shape) > 1:
        raise TypeError("`bbox` must be an 1-D array, got {}"
                        .format(bbox))
    if len(bbox) % 2 != 0:
        raise ValueError("`bbox` should have even number of elements, got {}"
                         .format(len(bbox)))

    ndim = len(bbox) // 2
    slices = [slice(bbox[d], bbox[d + ndim] + 1) for d in reversed(range(ndim))]
    return tuple(slices)


def bbox_to_shape(bbox):
    if not isinstance(bbox, (list, tuple, np.ndarray)):
        raise TypeError("`bbox` must be an iterable object, got {}"
                        .format(type(bbox)))
    if isinstance(bbox, np.ndarray) and len(bbox.shape) > 1:
        raise TypeError("`bbox` must be an 1-D array, got {}"
                        .format(bbox))
    if len(bbox) % 2 != 0:
        raise ValueError("`bbox` should have even number of elements, got {}"
                         .format(len(bbox)))

    ndim = len(bbox) // 2
    shape = [bbox[d + ndim] - bbox[d] + 1 for d in reversed(range(ndim))]
    return tuple(shape)


def extract_object(src_image, src_mask=None):
    """
    Extract a sub-volume of src_image where corresponding elements
    in src_mask is not zero.

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

    bbox = bbox_from_mask(src_mask, 1, padding=25)
    object_image = src_image[bbox_to_slices(bbox)]

    return object_image, bbox


def extract_region(mask, align=1, padding=0, min_bbox_shape=None):
    """ Extract a sub-region. Final region need meet several conditions:

    1. The shape of the sub-region is an integral multiple of `align`
    2. The shape of the sub-region is not less than `min_shape`

    Parameters
    ----------
    mask: ndarray
        a binary mask to represent liver region
    align: int or list of int
        the returned image should be aligned with `align`, default 1(ignore align)
    padding: int or list of int
        number pixels of padding in the image border (x, y, z, ...)
    min_bbox_shape: list of int
        minimum bounding box shape, passed to bbox_from_mask()

    Returns
    -------
    post_bbox: final bounding box
    """
    pattern = "Length of `{:s}` should be match with dimension of the image."

    # change to binary mask
    mask = np.asarray(mask, np.bool)
    ndim = mask.ndim

    if isinstance(align, int):
        align = (align,) * ndim
    elif isinstance(align, (list, tuple)):
        assert len(align) == ndim, pattern.format("align")
    else:
        raise TypeError("`align` must be an integer or a list of integer, got {}"
                        .format(type(align)))
    align = np.array(align, dtype=np.int32)

    if min_bbox_shape is None:
        min_bbox_shape = (1, ) * ndim
    pre_bbox = bbox_from_mask(mask, mask_values=1, min_shape=min_bbox_shape[::-1])
    ctr = (pre_bbox[:ndim] + pre_bbox[ndim:]) / 2
    liver_shape = pre_bbox[ndim:] - pre_bbox[:ndim] + 1     # (x, y) / (x, y, z)

    needed_shape = np.ceil(liver_shape / align).astype(np.int32) * align
    point1 = np.maximum(0, np.int32(ctr - (needed_shape - 1) / 2))
    point2 = np.minimum(np.array(mask.shape)[::-1] - 1, point1 + needed_shape - 1)
    point1 = np.maximum(0, point1 - padding)
    point2 = np.minimum(np.array(mask.shape)[::-1] - 1, point2 + padding)
    post_bbox = np.concatenate((point1, point2), axis=0)

    return post_bbox


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
    empty_slices = np.all(src_image == empty_value, axis=tuple(axes))
    return empty_slices


def get_largest_component(inputs, rank, connectivity=1):
    """
    Extract the largest connected component in input array.

    Parameters
    ----------
    inputs: ndarray
        An N-D array. Type of the array will be converted to boolean internally.
    rank: int
        Passed to generate_binary_structure
    connectivity: int
        Passed to generate_binary_structure

    Returns
    -------
    An int8 array with the same size as `inputs`. Entries in largest component have value 1, else 0.
    Return zeros array if `inputs` is a zero array.

    """
    struct = ndi.morphology.generate_binary_structure(rank, connectivity)
    res = inputs.astype(bool)
    if np.count_nonzero(res) == 0:
        return np.zeros_like(inputs, dtype=np.int8)

    labeled_res, n_res = ndi.label(res, struct)
    areas = np.bincount(labeled_res.flat)[1:]   # without background
    arg_min = np.argsort(areas)
    return merge_labels(labeled_res, [-1, int(arg_min[-1]) + 1])


def compute_robust_moments(binary_image, isotropic=False):
    """
    Compute robust center and standard deviation of a binary image(0: background, 1: foreground).

    Support n-dimension array.

    Parameters
    ----------
    binary_image: ndarray
        Input a image
    isotropic: boolean
        Compute isotropic standard deviation or not.

    Returns
    -------
    center: ndarray
        A vector with dimension = `binary_image.ndim`. Median of all the points assigned 1.
    std_dev: ndarray
        A vector with dimension = `binary_image.ndim`. Standard deviation of all the points assigned 1.

    Notes
    -----
    All the points assigned 1 are considered as a single object.

    """
    ndim = binary_image.ndim
    index = np.nonzero(binary_image)
    points = np.asarray(index).astype(np.float32)
    if points.shape[1] == 0:
        return np.array([-1.0] * ndim, dtype=np.float32), np.array([-1.0] * ndim, dtype=np.float32)
    points = np.transpose(points)
    points = np.fliplr(points)
    center = np.median(points, axis=0)

    # Compute median absolute deviation(short for mad) to estimate standard deviation
    if isotropic:
        diff = np.linalg.norm(points - center, axis=1)
        mad = np.median(diff)
        mad = np.array([mad] * ndim)
    else:
        diff = np.absolute(points - center)
        mad = np.median(diff, axis=0)
    std_dev = 1.4826 * mad
    std_dev = np.maximum(std_dev, [5.0] * ndim)
    return center, std_dev


def get_gd_image_single_obj(labels, center_perturb=0.2, stddev_perturb=0.4, blank_prob=0, partial=False):
    """
    Get gaussian distribution image with some perturbation. All points assigned 1 are considered
    to be the same object.

    Support n-dimension array.

    Parameters
    ----------
    labels: ndarray
        A binary image
    center_perturb: float
        Position perturbation of central point
    stddev_perturb: float
        Scale perturbation of standard deviation
    blank_prob: float
        Probability of returning a blank image(which means no gaussian distribution guide)
    partial: bool
        For test mode. If true only first slice has spatial guide for each tumor

    Returns
    -------
    gd: ndarray
        Gaussian distribution image

    """
    labels = np.asarray(labels, dtype=np.float32)
    ndim = labels.ndim
    if partial and ndim != 3:
        raise ValueError("If `partial` is True, `labels` must have rank 3, but get {}".format(ndim))

    if not np.any(labels) or random.random() < blank_prob:
        # return a blank gd image
        return np.zeros(labels.shape)

    idx = 0
    if partial:
        idx = np.where(np.count_nonzero(labels, axis=(1, 2)) > 0)[0][0]
        obj_lab = labels[idx]
        obj_ndim = ndim - 1
    else:
        obj_lab = labels
        obj_ndim = ndim

    center, std = compute_robust_moments(obj_lab)
    center_p_ratio = np.random.uniform(-center_perturb, center_perturb, obj_ndim)
    center_p = center_p_ratio * std + center
    std_p_ratio = np.random.uniform(1.0 / (1 + stddev_perturb), 1.0 + stddev_perturb, obj_ndim)
    std_p = std_p_ratio * std

    coords = [np.arange(0, shape) for shape in obj_lab.shape]
    coords = np.stack(np.meshgrid(*coords, indexing="ij"), axis=-1)
    normalizer = 2 * (std_p * std_p)
    d = np.exp(-np.sum((coords - center_p[::-1]) ** 2 / normalizer[::-1], axis=-1))
    cur_gd = np.clip(d, 0, 1)

    if partial:
        gd = np.zeros_like(labels)
        gd[idx] = cur_gd
        return gd
    else:
        return cur_gd


def get_gd_image_multi_objs(labels, center_perturb=0.2, stddev_perturb=0.4, blank_prob=0,
                            connectivity=1, partial=False):
    """
    Get gaussian distribution image with some perturbation. Only connected points assigned 1
    are considered to be the same object.

    Support n-dimension array.

    Parameters
    ----------
    labels: ndarray
        An arbitrary binary image
    center_perturb: float
        Position perturbation of central point
    stddev_perturb: float
        Scale perturbation of standard deviation
    blank_prob: float
        Probability of returning a blank image(which means no gaussian distribution guide)
    connectivity: integer
        Passed to function generate_binary_structure()
    partial: bool
        For test mode. If true only first slice has spatial guide for each tumor

    Returns
    -------
    gd: ndarray
        Gaussian distribution image

    """
    labels = np.asarray(labels, dtype=np.float32)
    ndim = labels.ndim

    if not np.any(labels):
        # return a blank gd image
        return np.zeros(labels.shape)

    # Label image and find all objects
    disc = ndi.generate_binary_structure(ndim, connectivity=connectivity)
    labeled_image, num_obj = ndi.label(labels, structure=disc)
    obj_images = [labeled_image == n + 1 for n in range(num_obj)]

    # Compute moments for each object
    gds = [get_gd_image_single_obj(obj_image, center_perturb, stddev_perturb, blank_prob, partial)
           for obj_image in obj_images]

    # reduce is faster than np.sum when input is a python list
    merged_gd = functools.reduce(lambda x, y: np.maximum(x, y), gds)

    return merged_gd


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
