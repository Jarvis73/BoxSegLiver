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

from data_kits import build_lits_liver, build_body


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
    # build_body.convert_to_dataset("body", seed=1234,source_path='/home/yf/Slimed_NF')
    build_body.convert_to_liver_bounding_box("body", seed=1234, source_path='/home/yf/Slimed_NF', align=(1, 1, 1),
                                             padding=(1, 1, 1), folds_file="k_folds.txt")
    # build_body.convert_to_liver_bbox_group("body", seed=1234, source_path='/home/yf/Slimed_NF', align=(1, 1, 1),
    #                                        padding=(1, 1, 1), prefix="tumor-bb-triplet", group="triplet",
    #                                        folds_file="k_folds.txt")


if __name__ == "__main__":
    main()
