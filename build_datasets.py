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
    # build_lits_liver.convert_to_liver("trainval", keep_only_liver=True, seed=1234)
    build_lits_liver.convert_to_liver_bounding_box("trainval", keep_only_liver=False, seed=1234,
                                                   align=8, padding=(25, 25, 2))
    # build_lits_liver.convert_to_liver_bbox_triplet("trainval", keep_only_liver=False, seed=1234,
    #                                                align=8, padding=(25, 25, 2))
    # build_lits_liver.convert_to_liver_bbox_triplet("trainval", keep_only_liver=False, seed=1234,
    #                                                align=8, padding=(25, 25, 2), only_tumor=True,
    #                                                prefix="tumor-bbox-triplet")


if __name__ == "__main__":
    main()
