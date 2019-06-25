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
import tensorflow as tf
from collections import Iterable

from utils import array_kits
# noinspection PyUnresolvedReferences
from utils.nii_kits import nii_reader, nii_writer
# noinspection PyUnresolvedReferences
from utils.mhd_kits import mhd_reader, mhd_writer

FORMATS = [
    "mhd",
    "nii"
]


class ImageReader(object):
    """
    Helper class that provides TensorFlow image coding utilities.

    Parameters
    ----------
    image_type: type
        image type to output
    extend_channel: bool
        extend an extra channel dimension or not.
        Typically, True for image and False for label.
    """

    def __init__(self, image_type=np.int16, extend_channel=False):
        self._format = "mhd"
        self._type = image_type
        self._extend_channel = extend_channel
        self._reader = mhd_reader
        self._writer = mhd_writer
        self._decode = None
        self._meta = None
        self._filename = None
        self._indices = []

    @property
    def shape(self):
        return self._decode.shape

    @property
    def name(self):
        return str(self._filename)

    @property
    def format(self):
        return self._format

    @format.setter
    def format(self, new_format):
        self._check_format(new_format)

    @property
    def indices(self):
        return self._indices

    @indices.setter
    def indices(self, inds):
        max_ind = self._decode.shape[0] - 1
        self._indices.clear()
        for ind in inds:
            if 0 <= ind <= max_ind:
                self._indices.append(ind)
            else:
                raise ValueError("Out of range: {} -> [0, {}]".format(ind, max_ind))

    def _check_format(self, new_format):
        if new_format == self._format:
            return
        assert new_format in FORMATS, "Not supported image format: {}".format(new_format)
        self._format = new_format
        self._reader = eval(new_format + "_reader")
        self._writer = eval(new_format + "_writer")

    def data(self, idx=None):
        if self._decode is None:
            return bytes()
        else:
            image = self.image(idx)
            if image is None:
                raise ValueError("No data: file_name ({})".format(self._filename))
            return image.tobytes()

    def image(self, idx=None):
        if idx is None:
            return self._decode
        if isinstance(idx, (int, np.int32, np.int64)):
            return self._decode[idx]
        elif isinstance(idx, (list, tuple)):
            slices = []
            for i in idx:
                if i < 0 or i >= self.shape[0]:
                    slices.append(np.zeros(self.shape[1:], dtype=self._decode.dtype))
                else:
                    slices.append(self._decode[i])
            if self._extend_channel:
                return np.concatenate(slices, axis=-1)
            else:
                return np.stack(slices, axis=-1)

    def flipud(self):
        self._decode = np.flip(self._decode, axis=1)

    def transform(self, transformer):
        self._decode = transformer(self._decode)

    def image_crop(self, idx=None):
        if idx is None:
            return self._decode
        if isinstance(idx, (int, np.int32, np.int64)):
            return self._decode[idx]
        elif isinstance(idx, (list, tuple)):
            slices = []
            for i in idx:
                if i < 0 or i >= self.shape[0]:
                    slices.append(np.zeros(self.shape[1:], dtype=self._decode.dtype))
                else:
                    slices.append(self._decode[i])
            if self._extend_channel:
                return np.concatenate(slices, axis=-1)
            else:
                return np.stack(slices, axis=-1)

    def read(self, file_name, idx=None):
        self._filename = file_name
        self._check_format(str(file_name).split(".")[1])
        self._meta, self._decode = self._reader(file_name)
        self._decode = self._decode.astype(self._type, copy=False)
        if self._extend_channel:
            self._decode = self._decode[..., None]
        self._indices = list(range(self._decode.shape[0]))
        return self.image(idx)

    def header(self, file_name):
        self._filename = file_name
        self._check_format(str(file_name).split(".")[1])
        self._meta, _ = self._reader(file_name, only_meta=True)
        return self.Header(self._meta, self.format)

    def save(self, save_path, img_array, meta=None, fmt="mhd"):
        if self._meta is None and meta is None:
            raise ValueError("Missing meta information")
        if meta is None and self._meta is not None:
            meta = self._meta
        self._check_format(fmt)

        if fmt == "nii" and meta is not None:
            img_array = img_array.transpose((2, 1, 0))
            qform = meta.get_qform()
            if qform[0, 3] < 0:
                img_array = np.flipud(img_array)
            if qform[1, 3] <= 0:
                img_array = np.fliplr(img_array)
        self._writer(save_path, img_array, meta_info=meta)

    def transpose(self, dims):
        self._decode = self._decode.transpose(dims)

    class Header(object):
        def __init__(self, header, fmt):
            self.header = header
            self.format = fmt

        @property
        def spacing(self):
            if self.format == "mhd":
                return tuple(self.header["ElementSpacing"])
            elif self.format == "nii":
                return self.header["srow_x"][0], self.header["srow_y"][1], self.header["srow_z"][2]

        @property
        def shape(self):
            if self.format == "mhd":
                return tuple(self.header["NDims"])
            elif self.format == "nii":
                return self.header.get_data_shape()[::-1]


class SubVolumeReader(ImageReader):
    """
    Add `bbox` attribute: [x1, y1, z1, ..., x2, y2, z2]
    """

    def __init__(self, image_type=np.int16, extend_channel=False):
        super(SubVolumeReader, self).__init__(image_type, extend_channel)
        self._bbox = []

    @property
    def bbox(self):
        return self._bbox

    @bbox.setter
    def bbox(self, bbox_):
        if not isinstance(bbox_, (list, tuple)):
            raise TypeError("`bbox` must be an iterable object, got {}"
                            .format(type(bbox_)))
        if len(bbox_) % 2 != 0:
            raise ValueError("`bbox` should have even number of elements, got {}"
                             .format(len(bbox_)))

        self._bbox = bbox_
        if self._decode is not None:
            self._clip_image()

    def _clip_image(self):
        ndim = self._decode.ndim - 1 if self._extend_channel else self._decode.ndim
        if len(self._bbox) // 2 != ndim:
            raise ValueError("Mismatched dimensions: self.bbox({}) vs self.image({})"
                             .format(len(self._bbox) // 2, ndim))

        self._decode = self._decode[array_kits.bbox_to_slices(self._bbox) +
                                    ((slice(None, None),) if self._extend_channel else ())]
        self._indices = list(range(self._decode.shape[0]))

    def read(self, file_name, idx=None):
        self._filename = file_name
        self._check_format(str(file_name).split(".")[-1])
        self._meta, self._decode = self._reader(file_name)
        self._decode = self._decode.astype(self._type, copy=False)

        if self._extend_channel:
            self._decode = self._decode[..., None]
        self._indices = list(range(self._decode.shape[0]))
        return self.image(idx)


def _int64_list_feature(values):
    """
    Return a TF-Feature of int64_list.

    Parameters
    ----------
    values: A scalar or list of values.

    Returns
    -------
    A TF-Feature.
    """
    if not isinstance(values, Iterable):
        values = [values]

    return tf.train.Feature(int64_list=tf.train.Int64List(
        value=values
    ))


def _float_list_feature(values):
    """
    Return a TF-Feature of float_list.

    Parameters
    ----------
    values: A scalar or list of values.

    Returns
    -------
    A TF-Feature.
    """
    if not isinstance(values, Iterable):
        values = [values]

    return tf.train.Feature(float_list=tf.train.FloatList(
        value=values
    ))


def _bytes_list_feature(values):
    """
    Return a TF-Feature of bytes

    Parameters
    ----------
    values: str or bytes

    Returns
    -------
    A TF-Feature

    """

    def to_bytes(value):
        if isinstance(value, str):
            return value.encode()
        elif isinstance(value, bytes):
            return value
        else:
            raise TypeError("Only str and bytes are supported, got {}"
                            .format(type(value)))

    return tf.train.Feature(bytes_list=tf.train.BytesList(
        value=[to_bytes(values)]
    ))


def _check_extra_info_type(extra_info):
    if extra_info is None:
        return {}, {}
    if not isinstance(extra_info, dict):
        raise TypeError("`extra_info` must be a dict, got {}".format(type(extra_info)))

    extra_split, extra_origin = {}, {}
    for key, val in extra_info.items():
        p = key.split("_")
        if len(p) < 2 or p[-1] not in ["split", "origin"]:
            raise ValueError("Keys in `extra_info` should have format 'xxx_split' or 'xxx_origin'")
        tag = "_".join(p[:-1])
        if p[-1] == "split":
            extra_split[tag] = val
        else:  # origin
            extra_origin[tag] = val

    return extra_split, extra_origin


def image_to_nf_examples(image_reader, label_reader, split=False, extra_str_info=None, extra_int_info=None,
                         group="triplet", group_label=False, mask_image=False, extra_float_info=None, patch_number=5):
    """ For NF data """
    if group not in ["none", "triplet", "quintuplet"]:
        raise ValueError("group must be one of [none, triplet, quintuplet], got {}".format(group))

    if image_reader.name is None:
        raise RuntimeError("ImageReader need call `read()` first.")
    extra_str_split, extra_str_origin = _check_extra_info_type(extra_str_info)
    extra_int_split, extra_int_origin = _check_extra_info_type(extra_int_info)
    extra_float_split, extra_float_origin = _check_extra_info_type(extra_float_info)

    if split:
        num_slices = image_reader.shape[0]
        for extra_list in extra_str_split.values():
            assert num_slices == len(extra_list), "Length not equal: {} vs {}".format(num_slices, len(extra_list))
        for extra_list in extra_int_split.values():
            assert num_slices == len(extra_list), "Length not equal: {} vs {}".format(num_slices, len(extra_list))
        for extra_list in extra_float_split.values():
            assert num_slices == len(extra_list), "Length not equal: {} vs {}".format(num_slices, len(extra_list))

        for idx in range(num_slices):
            if group == "triplet":
                indices = (idx - 1, idx, idx + 1)
            elif group == "quintuplet":
                indices = (idx - 2, idx - 1, idx, idx + 1, idx + 2)
            else:
                indices = idx
            if not mask_image:
                slices = np.array(image_reader.image(indices))
                slices_label = np.array(label_reader.image(indices))
            else:
                slices = np.array((image_reader.image(indices) * np.clip(label_reader.image(indices), 0, 1)
                                   .astype(image_reader.image().dtype)))
                slices_label = np.array(label_reader.image(indices))
            padding = (copy_number - slices.shape[1] % copy_number) % copy_number
            slices = np.pad(slices, ((0, 0), (0, padding), (0, 0)), 'constant')
            slices_label = np.pad(slices_label, ((0, 0), (0, padding), (0, 0)), 'constant')
            # Merge 2 patches to a larger one
            patch_height = (slices.shape[1] + padding) // copy_number

            # copy_number - 1 because merge 2 adjacent patches to a single patch
            for offset_index in range(copy_number - 1):
                slices_patch = slices[:, offset_index * patch_height:(offset_index + 2) * patch_height]
                slices_label_patch = slices_label[:, offset_index * patch_height:(offset_index + 2) * patch_height]
                feature_dict = {
                    "image/encoded": _bytes_list_feature(slices_patch.tobytes()),
                    "image/name": _bytes_list_feature(image_reader.name),
                    "image/format": _bytes_list_feature(image_reader.format),
                    "image/shape": _int64_list_feature(slices_patch.shape),
                    "segmentation/encoded": _bytes_list_feature(slices_label_patch.tobytes()),
                    "segmentation/name": _bytes_list_feature(label_reader.name),
                    "segmentation/format": _bytes_list_feature(label_reader.format),
                    "segmentation/shape": _int64_list_feature(slices_label_patch.shape),
                    "extra/number": _int64_list_feature(idx),
                    "extra/offset_height": _int64_list_feature(offset_index * patch_height),
                }
                for key, val in extra_str_split.items():
                    feature_dict["extra/{}".format(key)] = _bytes_list_feature(val[idx])
                # for key, val in extra_int_split.items():
                #     feature_dict["extra/{}".format(key)] = _int64_list_feature(val[idx])
                for key, val in extra_float_split.items():
                    feature_dict["extra/{}".format(key)] = _float_list_feature(val[idx])

                yield tf.train.Example(features=tf.train.Features(feature=feature_dict))
    else:
        feature_dict = {
            "image/encoded": _bytes_list_feature(image_reader.data()),
            "image/name": _bytes_list_feature(image_reader.name),
            "image/format": _bytes_list_feature(image_reader.format),
            "image/shape": _int64_list_feature(image_reader.shape),
            "segmentation/encoded": _bytes_list_feature(label_reader.data()),
            "segmentation/name": _bytes_list_feature(label_reader.name),
            "segmentation/format": _bytes_list_feature(label_reader.format),
            "segmentation/shape": _int64_list_feature(label_reader.shape)
        }

        for key, val in extra_str_origin.items():
            feature_dict["extra/{}".format(key)] = _bytes_list_feature(val)
        for key, val in extra_int_origin.items():
            feature_dict["extra/{}".format(key)] = _int64_list_feature(val)
        for key, val in extra_float_origin.items():
            feature_dict["extra/{}".format(key)] = _float_list_feature(val)

        yield tf.train.Example(features=tf.train.Features(feature=feature_dict))

