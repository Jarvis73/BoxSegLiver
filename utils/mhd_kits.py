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
import zlib
import numpy as np
from pathlib import Path
from collections import Iterable
from collections import OrderedDict as ODict


__all__ = [
    "MyList",
    "mhd_reader",
    "mhd_writer",
    "mhd_3dto2d",
    "METTypeRev"
]

METType = {
    'MET_CHAR': np.char,
    'MET_SHORT': np.int16,
    'MET_LONG': np.int32,
    'MET_INT': np.int32,
    'MET_UCHAR': np.uint8,
    'MET_USHORT': np.uint16,
    'MET_ULONG': np.uint32,
    'MET_UINT': np.uint32,
    'MET_FLOAT': np.float32,
}

METTypeRev = {
    np.dtype(np.int8): 'MET_CHAR',
    np.dtype(np.int16): 'MET_SHORT',
    np.dtype(np.int32): 'MET_INT',
    np.dtype(np.uint8): 'MET_UCHAR',
    np.dtype(np.uint16): 'MET_USHORT',
    np.dtype(np.uint32): 'MET_UINT',
    np.dtype(np.float32): 'MET_FLOAT',
    np.dtype(np.float64): 'MET_FLOAT',
}


class MyList(list):
    """ Reimplementing a list with new __repr__
    """
    def __init__(self, iterable):
        if not isinstance(iterable, Iterable):
            iterable = [iterable]
        self._obj = iterable
        super(MyList, self).__init__(iterable)

    def __repr__(self):
        str_list = [str(x) for x in self._obj]
        return " ".join(str_list)

    def __setitem__(self, key, value):
        self._obj[key] = value
        super(MyList, self).__setitem__(key, value)

    def __str__(self):
        return self.__repr__()

    def __format__(self, fmt_spec=""):
        return self.__repr__()


def mhd_reader(mhd_path, only_meta=False, compress=False, ranges=None, readonly=False):
    """ Implementation of `.mhd` file reader

    Parameters
    ----------
    mhd_path: str or Path
        file path to a mhd file
    only_meta: bool
        if True, raw image will not be loaded and the second return is None
    compress: bool
        if True, reader will decompress file first
    ranges: list or tuple
        two-element array, specify the slice range to read
    readonly: bool
        return a readonly array or not

    Returns
    -------
    meta_info: dict
        a dictionary contains all the information in mhd file
    raw_image: ndarray
        raw data of this image. If `only_meta` is True, this return will be None.

    """
    meta_info = ODict()
    mhd_path = Path(mhd_path)
    # read .mhd file
    with mhd_path.open() as f_mhd:
        for line in f_mhd.readlines():
            parts = line.split()
            meta_info[parts[0]] = ' '.join(parts[2:])

    primary_keys = ['NDims', 'DimSize', 'ElementType', 'ElementSpacing', 'ElementDataFile']
    for key in primary_keys:
        if key not in meta_info:
            print(meta_info)
            raise KeyError("Missing key `{}` in meta data of the mhd file".format(key))

    meta_info['NDims'] = int(meta_info['NDims'])
    meta_info['DimSize'] = MyList([eval(ele) for ele in meta_info['DimSize'].split()])
    meta_info['ElementSpacing'] = MyList([eval(ele) for ele in meta_info['ElementSpacing'].split()])

    raw_image = None
    if not only_meta:
        raw_path = mhd_path.with_name(meta_info['ElementDataFile'])

        # read .raw file
        if ranges is None:
            buffer = raw_path.read_bytes()
            new_shape = meta_info['DimSize'][::-1]
        else:
            z1, z2 = ranges
            with open(str(raw_path), "rb") as f:
                single_slice = meta_info["DimSize"][0] * meta_info["DimSize"][1] * 2
                f.seek(z1 * single_slice)
                buffer = f.read((z2 - z1) * single_slice)
            new_shape = [z2 - z1] + meta_info["DimSize"][-2::-1]

        if compress:
            buffer = zlib.decompress(buffer)
        raw_image = np.frombuffer(buffer, dtype=METType[meta_info['ElementType']])
        # if not readonly:
        #     raw_image.setflags(write=1)
        raw_image = np.reshape(raw_image, new_shape)

    return meta_info, raw_image


def mhd_writer(mhdpath, image, meta_info, verbose=False, compress=False):
    """ Implementation of `.mhd` file writer

    Parameters
    ----------
    mhdpath: str or Path
         file path to write at
    image: ndarray
        image to write
    meta_info: dict
        meta information. Note that value list should use class MyList
    verbose: bool
        print information or not
    compress: bool
        save compressed image or not
    """
    mhdpath = Path(mhdpath)
    with mhdpath.open("w") as fid:
        fid.write("NDims = {}\n".format(meta_info["NDims"]))
        fid.write("DimSize = {}\n".format(meta_info["DimSize"]))
        fid.write("ElementType = {}\n".format(meta_info["ElementType"]))
        fid.write("ElementSpacing = {}\n".format(meta_info["ElementSpacing"]))
        fid.write("ElementByteOrderMSB = {}\n".format(meta_info["ElementByteOrderMSB"]))
        fid.write("ElementDataFile = {}\n".format(meta_info["ElementDataFile"]))
        primary_keys = ['NDims', 'DimSize', 'ElementType', 'ElementSpacing',
                        'ElementDataFile', 'ElementByteOrderMSB']
        for key in meta_info:
            if key in primary_keys:
                continue
            fid.write("{} = {}\n".format(key, meta_info[key]))

    raw_path = mhdpath.parent / meta_info["ElementDataFile"]
    with raw_path.open("wb") as fid:
        if compress:
            fid.write(zlib.compress(image.astype(np.int16).tobytes(), zlib.Z_BEST_COMPRESSION))
        else:
            image.astype(np.int16).tofile(fid)

    if verbose:
        print("Write mhd to ", mhdpath)


def mhd_3dto2d(src_file, dst_dir, name_pattern="{}_{}"):
    meta, image = mhd_reader(src_file)
    assert list(meta["DimSize"]) == list(image.shape[:0:-1])
    meta["NDims"] = 2
    meta["DimSize"] = MyList(meta["DimSize"][:-1])
    meta["ElementSpacing"] = MyList(list(meta["ElementSpacing"])[:-1])
    name = meta["ElementDataFile"][:-4]
    for ind in range(image.shape[0]):
        meta["ElementDataFile"] = (name_pattern + ".raw").format(name, ind)
        mhd_writer(Path(dst_dir) / (name_pattern + ".mhd").format(name, ind), image[ind], meta)
