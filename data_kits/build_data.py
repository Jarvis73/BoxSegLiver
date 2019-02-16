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
# noinspection PyUnresolvedReferences
from utils.nii_kits import nii_reader
# noinspection PyUnresolvedReferences
from utils.mhd_kits import mhd_reader

FORMATS = [
    "mhd",
    "nii"
]


class ImageReader(object):
    """
    Helper class that provides TensorFlow image coding utilities.

    Parameters
    ----------
    image_format: str
        image format, supported formats are listed in FORMATS
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
        self._decode = None
        self._meta = None
        self._filename = None

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

    def _check_format(self, new_format):
        if new_format == self._format:
            return
        assert new_format in FORMATS, "Not supported image format: {}".format(new_format)
        self._format = new_format
        self._reader = eval(new_format + "_reader")

    def data(self, idx=None):
        if self._decode is None:
            return bytes()
        else:
            return self._decode.tobytes() if idx is None else self._decode[idx].tobytes()

    def image(self, idx=None):
        return self._decode if idx is None else self._decode[idx]

    def read(self, file_name, idx=None):
        self._filename = file_name
        self._check_format(str(file_name).split(".")[-1])
        self._meta, self._decode = self._reader(file_name)
        self._decode = self._decode.astype(self._type, copy=False)
        if self._extend_channel:
            self._decode = self._decode[..., None]
        return self.image(idx)

    def header(self, file_name):
        self._filename = file_name
        self._check_format(str(file_name).split(".")[-1])
        self._meta, _ = self._reader(file_name, only_meta=True)
        return self.Spacing(self._meta, self.format)

    class Spacing(object):
        def __init__(self, header, fmt):
            self.header = header
            self.format = fmt

        def spacing(self):
            if self.format == "mhd":
                return tuple(self._meta["ElementSpacing"])
            elif self.format == "nii":
                return self._meta["srow_x"][0], self._meta["srow_y"][1], self._meta["srow_z"][2]


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


def _bytes_list_feature(values):
    """
    Return a TF-Feature of bytes

    Parameters
    ----------
    values: str

    Returns
    -------
    A TF-Feature

    """
    def tobytes(value):
        return value.encode() if isinstance(value, str) else value

    return tf.train.Feature(bytes_list=tf.train.BytesList(
        value=[tobytes(values)]
    ))


def image_to_examples(image_reader, label_reader, split=False, extra_str_info=None, extra_int_info=None):
    """
    Convert N-D image and label to N-D/(N-1)-D tfexamples.

    This is a generator.

    Parameters
    ----------
    image_reader: ImageReader
        Containing image
    label_reader: ImageReader
        Containing label
    split: bool
        If true, image and label will be converted slice by slice.
    extra_str_info: dict
        extra information added to tfexample. Dict keys will be the tf-feature keys,
        and values must be a list whose length is equal with image_reader.shape[0]
    extra_int_info: dict

    Returns
    -------
    A list of TF-Example

    """
    if image_reader.name is None:
        raise RuntimeError("ImageReader need call `read()` first.")
    if extra_str_info is None:
        extra_str_info = {}
    if extra_int_info is None:
        extra_int_info = {}

    if split:
        num_slices = image_reader.shape[0]
        for extra_list in extra_str_info.values():
            assert num_slices == len(extra_list), "Length not equal: {} vs {}".format(num_slices, len(extra_list))
        for extra_list in extra_int_info.values():
            assert num_slices == len(extra_list), "Length not equal: {} vs {}".format(num_slices, len(extra_list))

        for idx in range(image_reader.shape[0]):
            feature_dict = {
                "image/encoded": _bytes_list_feature(image_reader.data(idx)),
                "image/name": _bytes_list_feature(image_reader.name),
                "image/format": _bytes_list_feature(image_reader.format),
                "image/shape": _int64_list_feature(image_reader.shape[1:]),
                "segmentation/encoded": _bytes_list_feature(label_reader.data(idx)),
                "segmentation/name": _bytes_list_feature(label_reader.name),
                "segmentation/format": _bytes_list_feature(label_reader.format),
                "segmentation/shape": _int64_list_feature(label_reader.shape[1:]),
                "extra/number": _int64_list_feature(idx),
            }
            for key, val in extra_str_info.items():
                feature_dict["extra/{}".format(key)] = _bytes_list_feature(str(val[idx]))
            for key, val in extra_int_info.items():
                feature_dict["extra/{}".format(key)] = _int64_list_feature(int(val[idx]))

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

        yield tf.train.Example(features=tf.train.Features(feature=feature_dict))
