import pickle
import numpy as np
import tensorflow as tf
from pathlib import Path

import data_kits.build_data as bd
from data_kits.build_data import ImageReader
from data_kits.build_data import SubVolumeReader
from data_kits.build_data import image_to_examples
from data_kits.build_data import bbox_to_examples
from data_kits.preprocess import random_split_k_fold
from utils import array_kits
from utils import misc
from utils import nii_kits


def get_body_list(source_path):
    # 硬编码了
    image_files = list(Path(source_path).rglob('STIR.mhd'))
    for i in range(len(image_files)):
        image_files[i] = str(image_files[i]).replace('stir', 'STIR')  # 由于我的笔记本读出来会小写，所以加了这个操作
    return image_files


def read_or_create_k_folds(path, file_names, k_split=None, seed=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        k_folds = []
        with path.open() as f:
            for line in f.readlines():
                k_folds.append([x for x in line[line.find(":") + 1:].strip().split(" ")])

    else:
        if not isinstance(k_split, int) or k_split <= 0:
            raise ValueError("Wrong `k_split` value. Need a positive integer, got {}".format(k_split))
        if k_split < 1:
            raise ValueError("k_split should >= 1, but get {}".format(k_split))
        k_folds = random_split_k_fold(file_names, k_split, seed) if k_split > 1 else [file_names]

        with path.open("w") as f:
            for i, fold in enumerate(k_folds):
                f.write("Fold %d:" % i)
                write_str = " ".join([str(x) for x in fold])
                f.write(write_str + "\n")
    return k_folds


def _get_body_records_dir(dataset):
    body_records = Path(__file__).parent.parent / 'data' / dataset / 'records'
    body_records.mkdir(parents=True, exist_ok=True)
    return body_records


def convert_to_dataset(dataset, source_path, k_split=5, seed=None, folds_file="k_folds.txt"):
    file_names = get_body_list(source_path)
    num_images = len(file_names)

    k_folds = read_or_create_k_folds(Path(__file__).parent.parent / 'data' / dataset / folds_file, file_names, k_split,
                                     seed)
    body_records = _get_body_records_dir(dataset)

    image_reader = ImageReader(np.int16, extend_channel=True)
    label_reader = ImageReader(np.uint8, extend_channel=False)  # use uint8 to save space

    # Convert each split
    counter = 1
    for i, fold in enumerate(k_folds):
        # Split to 2D slices for training
        output_filename_2d = body_records / "{}-2D-{}-of-{}.tfrecord".format(dataset, i + 1, k_split)
        # 3D volume for evaluation and prediction
        output_filename_3d = body_records / "{}-3D-{}-of-{}.tfrecord".format(dataset, i + 1, k_split)
        with tf.io.TFRecordWriter(str(output_filename_2d)) as writer_2d:
            with tf.io.TFRecordWriter(str(output_filename_3d)) as writer_3d:
                for j, image_file in enumerate(fold):
                    print("\r>> Converting fold {}, {}/{}, {}/{}".format(i + 1, j + 1, len(fold), counter, num_images),
                          end="")
                    # Read image
                    image_reader.read(image_file)
                    image_reader.flipud()
                    image_reader.transpose((1, 0, 2, 3))
                    seg_file = Path(str(image_file).replace("STIR.mhd", "STIR-label.mhd"))
                    label_reader.read(seg_file)
                    label_reader.flipud()
                    label_reader.transpose((1, 0, 2))
                    # we have extended extra dimension for image
                    if image_reader.shape[:-1] != label_reader.shape:
                        raise RuntimeError("Shape mismatched between image and label: {} vs {}".format(
                            image_reader.shape, label_reader.shape))
                    # Find empty slices(we skip them in training)
                    # extra_info = {"empty_split": array_kits.find_empty_slices(label_reader.image())}
                    # Convert 2D slices to example
                    for example in cute_image_to_examples(image_reader, label_reader, split=True):
                        writer_2d.write(example.SerializeToString())
                    # Convert 3D volume to example
                    for example in cute_image_to_examples(image_reader, label_reader, split=False):
                        writer_3d.write(example.SerializeToString())
                    counter += 1
                print()


def convert_to_liver_bounding_box(dataset, source_path, k_split=5, seed=None, align=1, padding=0, min_bbox_shape=None,
                                  prefix="bbox", folds_file="k_folds.txt"):
    """
    Convert dataset list to several tf-record files.

    Strip boundary of the CT images.
    """
    file_names = get_body_list(source_path)
    num_images = len(file_names)

    k_folds = read_or_create_k_folds(Path(__file__).parent.parent / 'data' / dataset / folds_file, file_names, k_split,
                                     seed)
    body_records = _get_body_records_dir(dataset)

    image_reader = SubVolumeReader(np.int16, extend_channel=True)
    label_reader = SubVolumeReader(np.uint8, extend_channel=False)  # use uint8 to save space

    # Convert each split
    counter = 1
    for i, fold in enumerate(k_folds):
        # Split to 2D slices for training
        output_filename_2d = body_records / "{}-{}-2D-{}-of-{}.tfrecord".format(dataset, prefix, i + 1, k_split)
        # 3D volume for evaluation and prediction
        output_filename_3d = body_records / "{}-{}-3D-{}-of-{}.tfrecord".format(dataset, prefix, i + 1, k_split)
        with tf.io.TFRecordWriter(str(output_filename_2d)) as writer_2d, \
                tf.io.TFRecordWriter(str(output_filename_3d)) as writer_3d:
            for j, image_file in enumerate(fold):
                print("\r>> Converting fold {}, {}/{}, {}/{}".format(i + 1, j + 1, len(fold), counter, num_images),
                      end="")

                # Read image
                seg_file = Path(str(image_file).replace("STIR.mhd", "STIR-label.mhd"))
                label_reader.read(seg_file)
                label_reader.flipud()
                label_reader.transpose((1, 0, 2))
                bbox = array_kits.extract_region(label_reader.image(), align, padding, min_bbox_shape)
                label_reader.bbox = bbox.tolist()
                image_reader.read(image_file)
                image_reader.flipud()
                image_reader.transpose((1, 0, 2, 3))
                image_reader.bbox = bbox.tolist()
                print(" {}".format(image_reader.shape), end="")
                if not np.all(np.array(image_reader.shape)[:3] % align[::-1] == 0):
                    raise ValueError("{}: box {} shape {}".format(image_file, bbox, image_reader.shape))

                # we have extended extra dimension for image
                if image_reader.shape[:-1] != label_reader.shape:
                    raise RuntimeError("Shape mismatched between image and label: {} vs {}".format(image_reader.shape,
                                                                                                   label_reader.shape))
                #
                extra_info = {"bbox_origin": bbox}
                # Convert 2D slices to example
                for example in cute_image_to_examples(image_reader, label_reader, split=True):
                    writer_2d.write(example.SerializeToString())
                    # Convert 3D volume to example
                for example in cute_image_to_examples(image_reader, label_reader, split=False,
                                                      extra_int_info=extra_info):
                    writer_3d.write(example.SerializeToString())
                counter += 1
            print()


def convert_to_liver_bbox_group(dataset,
                                source_path,
                                k_split=5,
                                seed=None,
                                align=1,
                                padding=0,
                                min_bbox_shape=None,
                                prefix="bbox-none",
                                group="none",
                                only_tumor=False,
                                mask_image=False,
                                folds_file="k_folds.txt"):
    file_names = get_body_list(source_path)
    num_images = len(file_names)

    k_folds = read_or_create_k_folds(Path(__file__).parent.parent / 'data' / dataset / folds_file, file_names, k_split,
                                     seed)
    body_records = _get_body_records_dir(dataset)

    image_reader = SubVolumeReader(np.int16, extend_channel=True)
    label_reader = SubVolumeReader(np.uint8, extend_channel=False)  # use uint8 to save space

    tumor_slices = []
    # Convert each split
    counter = 1
    print("Group: {}".format(group))
    for i, fold in enumerate(k_folds):
        # Split to 2D slices for training
        output_filename_2d = body_records / "{}-{}-2D-{}-of-{}.tfrecord".format(dataset, prefix, i + 1, k_split)
        print("Write to {}".format(str(output_filename_2d)))
        with tf.io.TFRecordWriter(str(output_filename_2d)) as writer_2d:
            for j, image_file in enumerate(fold):
                print("\r>> Converting fold {}, {}/{}, {}/{}".format(i + 1, j + 1, len(fold), counter, num_images),
                      end="")

                # Read image
                seg_file = Path(str(image_file).replace("STIR.mhd", "STIR-label.mhd"))
                label_reader.read(seg_file)
                label_reader.flipud()
                label_reader.transpose((1, 0, 2))
                bbox = array_kits.extract_region(np.squeeze(label_reader.image()), align, padding, min_bbox_shape)
                label_reader.bbox = bbox.tolist()
                image_reader.read(image_file)
                image_reader.flipud()
                image_reader.transpose((1, 0, 2, 3))
                image_reader.bbox = bbox.tolist()
                print(" {}".format(image_reader.shape), end="")
                if not np.all(np.array(image_reader.shape)[:3] % align[::-1] == 0):
                    raise ValueError("{}: box {} shape {}".format(image_file.stem, bbox, image_reader.shape))

                if only_tumor:
                    # Extract tumor slices
                    tumor_value = np.max(label_reader.image())
                    indices = np.where(np.max(label_reader.image(), axis=(1, 2)) == tumor_value)[0]
                    image_reader.indices = indices
                    label_reader.indices = indices
                    print("  #Tumor slices: {:d}   ".format(len(indices)), end="", flush=True)
                    tumor_slices.extend(indices)

                # we have extended extra dimension for image
                if image_reader.shape[:-1] != label_reader.shape:
                    raise RuntimeError("Shape mismatched between image and label: {} vs {}".format(
                        image_reader.shape, label_reader.shape))
                # Convert 2D slices to example
                for example in image_to_examples(image_reader, label_reader, split=True,
                                                 group=group, group_label=only_tumor,
                                                 mask_image=mask_image):
                    writer_2d.write(example.SerializeToString())
                counter += 1
            print()

    if only_tumor:
        print("Total #tumor slices: {}".format(len(tumor_slices)))


def dump_fp_bbox_from_prediction(label_dirs, pred_dir):
    pred_dir = Path(pred_dir)
    save_path = pred_dir.parent / "bboxes-{}.pkl".format(pred_dir.parent.name)

    all_bboxes = {}
    counter = 0
    for pred_path in sorted(pred_dir.glob("prediction-*.nii.gz")):
        print(pred_path.name)
        lab_file = pred_path.stem.replace("prediction", "segmentation")
        lab_path = misc.find_file(label_dirs, lab_file)
        result = array_kits.merge_labels(nii_kits.nii_reader(pred_path)[1], [0, 2])
        reference = array_kits.merge_labels(nii_kits.nii_reader(lab_path)[1], [0, 2])
        fps, tps = array_kits.find_tp_and_fp(result, reference)
        for x in fps:
            print(x, [x[3] - x[0], x[4] - x[1], x[5] - x[2]])
        print()
        counter += len(fps)
        for x in tps:
            print(x, [x[3] - x[0], x[4] - x[1], x[5] - x[2]])
        print("#" * 80)
        all_bboxes[int(pred_path.stem.replace(".nii", "").split("-")[-1])] = {
            "fps": fps, "tps": tps
        }
    print("FPs: {}".format(counter))

    with save_path.open("wb") as f:
        pickle.dump(all_bboxes, f, pickle.HIGHEST_PROTOCOL)


def convert_to_tp_dataset(dataset,
                          source_path,
                          k_split=5,
                          folds_file="k_folds.txt",
                          seed=None,
                          align=1,
                          padding=0,
                          min_bbox_shape=None,
                          prefix="cls-0tp"):
    file_names = get_body_list(source_path)
    num_images = len(file_names)

    k_folds = read_or_create_k_folds(Path(__file__).parent.parent / 'data' / dataset / folds_file, file_names, k_split,
                                     seed)
    body_records = _get_body_records_dir(dataset)

    label_reader = SubVolumeReader(np.uint8, extend_channel=False)

    counter = 1
    for i, fold in enumerate(k_folds):
        output_filename = body_records / "{}-{}-of-{}.tfrecord".format(prefix, i + 1, k_split)
        with tf.io.TFRecordWriter(str(output_filename)) as writer:
            for j, image_file in enumerate(fold):
                print("\r>> Converting fold {}, {}/{}, {}/{}".format(i + 1, j + 1, len(fold), counter, num_images),
                      end="")
                seg_file = Path(str(image_file).replace("STIR.mhd", "STIR-label.mhd"))
                label_reader.read(seg_file)
                label_reader.flipud()
                label_reader.transpose((1, 0, 2))
                bbox = array_kits.extract_region(label_reader.image(), align, padding, min_bbox_shape)
                label_reader.bbox = bbox.tolist()

                tumor_value = np.max(label_reader.image())
                indices = np.where(np.max(label_reader.image(), axis=(1, 2)) == tumor_value)[0]
                label_reader.indices = indices

                tps = array_kits.find_tp(array_kits.merge_labels(label_reader.image(), [0, 2]),
                                         split=True)

                # list of fps is sorted by z, y, x
                for example in bbox_to_examples(label_reader, tps):
                    writer.write(example.SerializeToString())
                counter += 1
            print()


def convert_to_classify_dataset(dataset,
                                source_path,
                                pred_dir,
                                prefix,
                                k_split=5,
                                seed=None,
                                align=1,
                                padding=0,
                                min_bbox_shape=None):
    pred_dir = Path(pred_dir)
    file_names = get_body_list(source_path)
    num_images = len(file_names)

    k_folds = read_or_create_k_folds(Path(__file__).parent.parent / "data/LiTS/k_folds.txt",
                                     file_names, k_split, seed)
    body_records = _get_body_records_dir(dataset)

    label_reader = SubVolumeReader(np.uint8, extend_channel=False)  # use uint8 to save space
    pred_reader = SubVolumeReader(np.uint8, extend_channel=False)

    counter = 1
    for i, fold in enumerate(k_folds):
        output_filename = body_records / "cls-{}-{}-of-{}.tfrecord".format(prefix, i + 1, k_split)
        with tf.io.TFRecordWriter(str(output_filename)) as writer:
            for j, image_file in enumerate(fold):
                print("\r>> Converting fold {}, {}/{}, {}/{}".format(i + 1, j + 1, len(fold), counter, num_images),
                      end="")

                seg_file = Path(str(image_file).replace("STIR.mhd", "STIR-label.mhd"))
                label_reader.read(seg_file)
                label_reader.flipud()
                label_reader.transpose((1, 0, 2))
                bbox = array_kits.extract_region(label_reader.image(), align, padding, min_bbox_shape)
                label_reader.bbox = bbox.tolist()

                pred_file = pred_dir / image_file.name.replace("volume", "prediction")
                pred_file = pred_file.with_suffix(".nii.gz")
                pred_reader.read(pred_file)
                pred_reader.bbox = bbox.tolist()

                tumor_value = np.max(label_reader.image())
                indices = np.where(np.max(label_reader.image(), axis=(1, 2)) == tumor_value)[0]
                label_reader.indices = indices

                fps, _ = array_kits.find_tp_and_fp(pred_reader.image(), label_reader.image())
                # list of fps is sorted by z, y, x
                for example in bbox_to_examples(label_reader, fps):
                    writer.write(example.SerializeToString())
                counter += 1
            print()


def hist_to_examples(label_reader, liver_hist_array, tumor_hist_array, split=False):
    assert len(liver_hist_array.shape) == 2
    assert len(tumor_hist_array.shape) == 2
    if split:
        liver_length = liver_hist_array.shape[1]
        tumor_length = tumor_hist_array.shape[1]
        for idx in label_reader.indices:
            feature_dict = {
                "liver_hist/encoded": bd._bytes_list_feature(liver_hist_array[idx].tobytes()),
                "liver_hist/shape": bd._int64_list_feature(liver_length),
                "tumor_hist/encoded": bd._bytes_list_feature(tumor_hist_array[idx].tobytes()),
                "tumor_hist/shape": bd._int64_list_feature(tumor_length),
                "case/name": bd._bytes_list_feature(label_reader.name)
            }
            yield tf.train.Example(features=tf.train.Features(feature=feature_dict))
    else:
        feature_dict = {
            "liver_hist/encoded": bd._bytes_list_feature(liver_hist_array.tobytes()),
            "liver_hist/shape": bd._int64_list_feature(liver_hist_array.shape),
            "tumor_hist/encoded": bd._bytes_list_feature(tumor_hist_array.tobytes()),
            "tumor_hist/shape": bd._int64_list_feature(tumor_hist_array.shape),
            "case/name": bd._bytes_list_feature(label_reader.name)
        }
        yield tf.train.Example(features=tf.train.Features(feature=feature_dict))


def cute_image_to_examples(image_reader,
                           label_reader,
                           cute_num=5,
                           split=False,
                           extra_str_info=None,
                           extra_int_info=None,
                           group="triplet",
                           group_label=False,
                           mask_image=False,
                           extra_float_info=None):
    if group not in ["none", "triplet", "quintuplet"]:
        raise ValueError("group must be one of [none, triplet, quintuplet], got {}".format(group))
    if image_reader.name is None:
        raise RuntimeError("ImageReader need call `read()` first.")
    extra_str_split, extra_str_origin = bd._check_extra_info_type(extra_str_info)
    extra_int_split, extra_int_origin = bd._check_extra_info_type(extra_int_info)
    extra_float_split, extra_float_origin = bd._check_extra_info_type(extra_float_info)
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
            slices = image_reader.image(indices)
            slices_label = label_reader.image(idx)
            padding = cute_num - slices.shape[0] % cute_num
            slices = np.pad(slices, ((padding // 2, padding - padding // 2), (0, 0), (0, 0)), 'constant')
            slices_label = np.pad(slices_label, ((padding // 2, padding - padding // 2), (0, 0)), 'constant')
            sub_slice_height = slices.shape[0] // cute_num
            for offset_index in range(cute_num - 1):
                slices = slices[offset_index * sub_slice_height:(offset_index + 2) * sub_slice_height]
                slices_label = slices_label[offset_index * sub_slice_height:(offset_index + 2) * sub_slice_height]
                feature_dict = {
                    "image/encoded": bd._bytes_list_feature(slices.tobytes()),
                    "image/name": bd._bytes_list_feature(image_reader.name),
                    "image/format": bd._bytes_list_feature(image_reader.format),
                    "image/shape": bd._int64_list_feature(slices.shape),
                    "segmentation/encoded": bd._bytes_list_feature(slices_label.tobytes()),
                    "segmentation/name": bd._bytes_list_feature(label_reader.name),
                    "segmentation/format": bd._bytes_list_feature(label_reader.format),
                    "segmentation/shape": bd._int64_list_feature(slices_label.shape),
                    "extra/number": bd._int64_list_feature(idx),
                    "extra/offset_height": bd._int64_list_feature(offset_index),
                    "extra/sub_slice_height": bd._int64_list_feature(sub_slice_height),
                    "extra/ud_padding": bd._int64_list_feature(padding),
                }
                for key, val in extra_str_split.items():
                    feature_dict["extra/{}".format(key)] = bd._bytes_list_feature(val[idx])
                    # for key, val in extra_int_split.items():
                    #     feature_dict["extra/{}".format(key)] = _int64_list_feature(val[idx])
                for key, val in extra_float_split.items():
                    feature_dict["extra/{}".format(key)] = bd._float_list_feature(val[idx])

                yield tf.train.Example(features=tf.train.Features(feature=feature_dict))
    else:
        image = image_reader.image()
        label = label_reader.image()
        padding = cute_num - image.shape[1] % cute_num
        image = np.pad(image, ((0, 0), (padding // 2, padding - padding // 2), (0, 0), (0, 0)), 'constant')
        label = np.pad(label, ((0, 0), (padding // 2, padding - padding // 2), (0, 0)), 'constant')
        sub_slice_height = image.shape[1] // cute_num
        for offset_index in range(cute_num - 1):
            slices = image[:, offset_index * sub_slice_height:(offset_index + 2) * sub_slice_height, :, :]
            slices_label = label[:, offset_index * sub_slice_height:(offset_index + 2) * sub_slice_height, :]
            feature_dict = {
                "image/encoded": bd._bytes_list_feature(slices.tobytes()),
                "image/name": bd._bytes_list_feature(image_reader.name),
                "image/format": bd._bytes_list_feature(image_reader.format),
                "image/shape": bd._int64_list_feature(slices.shape),
                "segmentation/encoded": bd._bytes_list_feature(slices_label.tobytes()),
                "segmentation/name": bd._bytes_list_feature(label_reader.name),
                "segmentation/format": bd._bytes_list_feature(label_reader.format),
                "segmentation/shape": bd._int64_list_feature(slices_label.shape),
                "extra/offset_height": bd._int64_list_feature(offset_index),
                "extra/sub_slice_height": bd._int64_list_feature(sub_slice_height),
                "extra/ud_padding": bd._int64_list_feature(padding),
            }

            for key, val in extra_str_origin.items():
                feature_dict["extra/{}".format(key)] = bd._bytes_list_feature(val)
            for key, val in extra_int_origin.items():
                feature_dict["extra/{}".format(key)] = bd._int64_list_feature(val)
            for key, val in extra_float_origin.items():
                feature_dict["extra/{}".format(key)] = bd._float_list_feature(val)

            yield tf.train.Example(features=tf.train.Features(feature=feature_dict))


def convert_to_histogram_dataset(dataset, k_split=5, seed=None, prefix="hist", folds_file="k_folds.txt", bins=100,
                                 xrng=(-200, 250)):
    file_names = get_body_list(dataset)
    num_images = len(file_names)

    k_folds = read_or_create_k_folds(Path(__file__).parent.parent / 'data' / dataset / folds_file, file_names, k_split,
                                     seed)
    body_records = _get_body_records_dir(dataset)

    image_reader = SubVolumeReader(np.int16, extend_channel=True)
    label_reader = SubVolumeReader(np.uint8, extend_channel=False)  # use uint8 to save space

    # Convert each split
    counter = 1
    for i, fold in enumerate(k_folds):
        output_filename_2d = body_records / "{}-{}-{}_{}-2D-{}-of-{}.tfrecord" \
            .format(prefix, bins, xrng[0], xrng[1], i + 1, k_split)
        output_filename_3d = body_records / "{}-{}-{}_{}-3D-{}-of-{}.tfrecord" \
            .format(prefix, bins, xrng[0], xrng[1], i + 1, k_split)
        print("Write to {} and {}".format(str(output_filename_2d), str(output_filename_3d)))
        with tf.io.TFRecordWriter(str(output_filename_2d)) as writer_2d, \
                tf.io.TFRecordWriter(str(output_filename_3d)) as writer_3d:
            for j, image_file in enumerate(fold):
                print("\r>> Converting fold {}, {}/{}, {}/{}".format(i + 1, j + 1, len(fold), counter, num_images),
                      end="")

                # Read image
                seg_file = Path(str(image_file).replace("STIR.mhd", "STIR-label.mhd"))
                label_reader.read(seg_file)
                label_reader.flipud()
                label_reader.transpose((1, 0, 2))
                image_reader.read(image_file)
                image_reader.flipud()
                image_reader.transpose((1, 0, 2, 3))

                # we have extended extra dimension for image
                if image_reader.shape[:-1] != label_reader.shape:
                    raise RuntimeError("Shape mismatched between image and label: {} vs {}".format(
                        image_reader.shape, label_reader.shape))

                images = image_reader.image()
                labels = label_reader.image()
                liver, tumor = images[labels == 1], images[labels == 2]
                val1, _ = np.histogram(liver.flat, bins=bins, range=xrng, density=True)
                val2, _ = np.histogram(tumor.flat, bins=bins, range=xrng, density=True)
                val1_total = np.tile(val1[None, :].astype(np.float32), [len(label_reader.indices), 1])
                val2_total = np.tile(val2[None, :].astype(np.float32), [len(label_reader.indices), 1])

                # Convert 2D slices to example
                for example in hist_to_examples(label_reader, val1_total, val2_total, split=True):
                    writer_2d.write(example.SerializeToString())
                # Convert 3D slices to example
                for example in hist_to_examples(label_reader, val1_total, val2_total, split=False):
                    writer_3d.write(example.SerializeToString())
                counter += 1
            print()
