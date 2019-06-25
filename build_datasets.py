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

from data_kits import build_lits_liver


def main():
    # Only liver
    # build_lits_liver.convert_to_liver("trainval", keep_only_liver=True, seed=1234)

    # Liver and tumor
    # build_lits_liver.convert_to_liver_bounding_box("trainval", keep_only_liver=False, seed=1234,
    #                                                align=(16, 16, 1), padding=(25, 25, 2),
    #                                                folds_file="k_folds.txt")

    # Liver and tumor --> triplet as input
    # build_lits_liver.convert_to_liver_bbox_group("trainval", keep_only_liver=False, seed=1234,
    #                                              align=(, 16, 1), padding=(25, 25, 2),
    #                                              prefix="bbox-triplet", group="triplet")

    # Only tumor --> triplet as input
    # build_lits_liver.convert_to_liver_bbox_group("trainval", keep_only_liver=False, seed=1234,
    #                                              align=(16, 16, 1), padding=(25, 25, 2),
    #                                              only_tumor=True,
    #                                              prefix="tumor-bb-triplet", group="triplet",
    #                                              folds_file="k_folds.txt")

    # Liver and tumor --> quintuplet as input
    # build_lits_liver.convert_to_liver_bbox_group("trainval", keep_only_liver=False, seed=1234,
    #                                              align=(16, 16, 1), padding=(25, 25, 2),
    #                                              prefix="bbox-quintuplet", group="quintuplet",
    #                                              folds_file="k_folds.txt")

    # Ground truth bounding box
    # build_lits_liver.convert_to_tp_dataset("trainval", align=(16, 16, 1), padding=(25, 25, 2),
    #                                        folds_file="k_folds.txt")

    # Histogram
    # build_lits_liver.convert_to_histogram_dataset("trainval", keep_only_liver=False, seed=1234,
    #                                               bins=100, xrng=(-200, 250),
    #                                               folds_file="k_folds.txt", guide=None, hist_scale="total")
    # build_lits_liver.convert_to_histogram_dataset("trainval", keep_only_liver=False, seed=1234,
    #                                               bins=100, xrng=(-200, 250), prefix="guide-hist",
    #                                               folds_file="k_folds.txt", guide="middle", hist_scale="total")

    # build_lits_liver.convert_to_histogram_dataset("trainval", keep_only_liver=False, seed=1234,
    #                                               align=(16, 16, 1), padding=(25, 25, 2),
    #                                               bins=100, xrng=(-200, 250), prefix="slice-hist",
    #                                               folds_file="k_folds.txt", guide="middle", hist_scale="slice")

    # Histogram v2
    # build_lits_liver.convert_to_histogram_dataset("trainval", keep_only_liver=False, seed=1234,
    #                                               bins=100, xrng=(-100, 250),
    #                                               folds_file="k_folds.txt", guide=None, hist_scale="total")

    #                                               folds_file="k_folds.txt")


def build_eval_guide(tfrecord):
    import tensorflow as tf
    from utils import array_kits
    import numpy as np
    import pickle
    from pathlib import Path
    dataset = tf.data.TFRecordDataset(tfrecord)

    def parse_fn(example):
        features = {
            "image/name": tf.FixedLenFeature([], tf.string),
            "segmentation/encoded": tf.FixedLenFeature([], tf.string),
            "segmentation/shape": tf.FixedLenFeature([3], tf.int64),
        }
        features = tf.parse_single_example(example, features=features)

        label = tf.decode_raw(features["segmentation/encoded"], tf.uint8, name="DecodeMask")
        label = tf.reshape(label, features["segmentation/shape"], name="ReshapeMask")
        label = tf.cast(label, tf.int32)

        return label, features["image/name"]

    dataset = dataset.map(parse_fn).make_initializable_iterator()
    labels, names = dataset.get_next()

    all_sp_guides = []
    with tf.Session() as sess:
        sess.run(dataset.initializer)
        while True:
            try:
                labels_val, name = sess.run([labels, names])
                sp_guide = array_kits.get_moments_multi_objs(labels_val, 2, partial=True, partial_slice="middle")
                print(name, sp_guide.shape)
                all_sp_guides.append(sp_guide.astype(np.float32))
            except tf.errors.OutOfRangeError:
                break

    save_path = Path(__file__).parent / "data/LiTS/test-3-of-5-mmts.pkl"
    with save_path.open("wb") as f:
        pickle.dump(all_sp_guides, f, pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()
    # build_eval_guide("/data/jarvis/TF_1_13_MedicalImageSegmentation/data/LiTS/records/trainval-bbox-3D-3-of-5.tfrecord")
