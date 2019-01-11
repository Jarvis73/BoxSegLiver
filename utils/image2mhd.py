# Copyright 2019 Zhang Jianwei All Right Reserved.
#
# TODO: Choice a License
#
# =================================================================================

import os
from pathlib import Path
from utils.mhd_kits import *
from utils.nii_kits import *


def vol2mhd(mode="3D", img="I", in_path=None, out_path=None, worker="VolConverter"):
    if not Path(in_path).exists():
        raise FileNotFoundError("Can not find", in_path)
    if mode not in ["3D", "2D"]:
        raise ValueError("Unsupported value of mode: {}, please choice from [3D, 2D]".format(mode))
    if img not in ["I", "M"]:
        raise ValueError("Unsupported value of img: {}, please choice from [I, M]".format(mode))
    os.system("{} {} {} {} {} > nul".format(worker, mode, img, in_path, out_path))


def nii2mhd(mode="3D", img="I", in_path=None, out_path=None, verbose=False):
    if not Path(in_path).exists():
        raise FileNotFoundError("Can not find", in_path)
    if img not in ["I", "M"]:
        raise ValueError("Unsupported value of img: {}, please choice from [I, M]".format(mode))

    header, image = nii_reader(in_path)
    if img == "M":
        image = image * 255

    if mode == "3D":
        meta_info = {"NDims": 3,
                     "DimSize": MyList(header.get_data_shape()),
                     "ElementType": METTypeRev[header["datatype"].dtype],
                     "ElementSpacing": MyList([abs(header['srow_x'][0]),
                                               abs(header['srow_y'][1]),
                                               abs(header['srow_z'][2])]),
                     "ElementByteOrderMSB": "False",
                     "ElementDataFile": str(Path(out_path).stem) + ".raw"
                     }
        mhd_writer(out_path + ".mhd", image, meta_info, verbose)
    elif mode == "2D":
        meta_info = {"NDims": 2,
                     "DimSize": MyList(header.get_data_shape()[:2]),
                     "ElementType": METTypeRev[header["datatype"].dtype],
                     "ElementSpacing": MyList([abs(header['srow_x'][0]),
                                               abs(header['srow_y'][1])]),
                     "ElementByteOrderMSB": "False",
                     }
        for i in range(header.get_data_shape()[2]):
            base_str = "{}_{}"
            mhd_path = base_str.format(out_path, i) + ".mhd"
            raw_path = base_str.format(str(Path(out_path).stem), i) + ".raw"
            meta_info["ElementDataFile"] = raw_path
            mhd_writer(mhd_path, image[i], meta_info, verbose)
    else:
        raise ValueError("Unsupported value of mode: {}, please choice from [3D, 2D]".format(mode))


def extract_data(src_dir, mode, dst_origin_dir, dst_mask_dir, origin, mask,
                 data_type="vol", name=None, exist_mask=False):
    """ Convert vol/nii data to mhd data.

    Parameters
    ----------
    src_dir: str or Path
        source directory
    mode: str
        2D/3D, extract 3D volume or 2D slices
    dst_origin_dir: str or Path
        destination directory to store origin image
    dst_mask_dir: str or Path
        destination directory to store mask image
    origin: str or Path
        path to original image, base on `src_dir`
    mask: str or Path
        path to mask image, base on `src_dir`
    data_type: str
        vol or nii source
    name: str
        uniform name for extracted images. If none, directory name will be used
    exist_mask: bool
        extract mask or not

    Keyword arguments
    -----------------
    Parameters passed to converter function.

    Example
    -------
    SrcDir = "D:/DataSet/LiverQL/Source"
    DstDir_o = "D:/DataSet/LiverQL/Destination/origin"
    DstDir_m = "D:/DataSet/LiverQL/Destination/mask"
    origin = "Study/Study_Phase2.vol"
    mask = "Study/Study_Phase2_Label.vol"
    extract_data(SrcDir, "3D", DstDir_o, DstDir_m, origin, mask, type="vol", name="S")

    SrcDir = "D:/DataSet/LiverQL/Source"
    DstDir_o = "D:/DataSet/LiverQL/Destination/origin"
    DstDir_m = "D:/DataSet/LiverQL/Destination/mask"
    origin = "volume-*.nii"
    mask = "segmentation-*.nii"
    extract_data(SrcDir, "3D", DstDir_o, DstDir_m, origin, mask, type="nii", name="S")
    """
    roots = [Path(x) for x in src_dir.split("+")]
    dst_origin_dir = Path(dst_origin_dir)
    dst_origin_dir.mkdir(parents=True, exist_ok=True)
    if exist_mask:
        dst_mask_dir = Path(dst_mask_dir)
        dst_mask_dir.mkdir(parents=True, exist_ok=True)

    i = 1
    for root in roots:
        if data_type == "vol":
            for src in root.iterdir():
                if not src.is_dir():
                    continue
                src_origin = src/origin

                print(i, src)
                if name:
                    dst = "{:s}{:03d}".format(name, i)
                else:
                    dst = src
                dst_origin = dst_origin_dir/((dst + "_o") if mode == "2D" else dst)
                vol2mhd(mode, "I", str(src_origin), str(dst_origin))

                if exist_mask:
                    src_mask = src / mask
                    dst_mask = dst_mask_dir/(dst + "_m")
                    vol2mhd(mode, "M", str(src_mask), str(dst_mask))
                i += 1
        elif data_type == "nii":
            origins = root.glob(origin)
            masks = root.glob(mask)
            for src_origin, src_mask in zip(sorted(origins), sorted(masks)):
                # Sorting array is required in linux system, but windows not
                print(i, src_origin.stem, src_mask.stem)
                if name:
                    dst = "{:s}{:03d}".format(name, i)
                else:
                    dst = src_origin.stem
                dst_origin = dst_origin_dir/((dst + "_o") if mode == "2D" else dst)
                dst_mask = dst_mask_dir/(dst + "_m")
                nii2mhd(mode, "I", str(src_origin), str(dst_origin))
                nii2mhd(mode, "M", str(src_mask), str(dst_mask))
                i += 1
        elif data_type == "nii_origin":
            origins = root.glob(origin)
            for src_origin in sorted(origins):
                # Sorting array is required in linux system, but windows not
                print(i, src_origin.stem)
                if name:
                    dst = "{:s}{:03d}".format(name, i)
                else:
                    dst = src_origin.stem
                dst_origin = dst_origin_dir / ((dst + "_o") if mode == "2D" else dst)
                nii2mhd(mode, "I", str(src_origin), str(dst_origin))
                i += 1
    return
