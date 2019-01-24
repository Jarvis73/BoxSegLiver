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
import argparse


def add_arguments(parser):
    group = parser.add_argument_group(title="Global Arguments")
    group.add_argument("--tag",
                       type=str,
                       required=True, help="Configuration tag(like UID)")

    group = parser.add_argument_group(title="Input Pipeline Arguments")
    group.add_argument("--w_width",
                       type=float,
                       default=450,
                       required=False, help="Medical image window width (default: %(default)d)")
    group.add_argument("--w_level",
                       type=float,
                       default=50,
                       required=False, help="Medical image window level (default: %(default)d)")
    group.add_argument("--zoom",
                       action="store_true",
                       required=False, help="Augment dataset with random zoom in and shift")
    group.add_argument("--zoom_scale",
                       type=float,
                       default=1.5,
                       required=False, help="Maximum random zoom-in scale. Make sure zoom_scale >= 1. "
                                            "(default: %(default)f)")
    group.add_argument("--noise",
                       action="store_true",
                       required=False, help="Augment dataset with random noise")
    group.add_argument("--noise_scale",
                       type=float,
                       default=0.05,
                       required=False, help="Random noise scale (default: %(default)f)")

    group = parser.add_argument_group(title="Model Arguments")
    group.add_argument("--mode",
                       type=str,
                       default="TRAIN",
                       choices=["TRAIN", "EVAL", "PREDICT"],
                       required=True, help="Model mode for train/val/test (default: %(default)s)")
    group.add_argument("--classes",
                       type=str,
                       nargs="+",
                       required=True, help="Class names of the objects")
    group.add_argument("--batch_size",
                       type=int,
                       default=8,
                       required=False, help="Model batch size (default: %(default)d)")
    group.add_argument("--weight_init",
                       type=str,
                       default="trunc_norm",
                       choices=["trunc_norm", "xavier"],
                       required=False, help="Model variable initialization method (default: %(default)s)")
    group.add_argument("--normalizer",
                       type=str,
                       default="batch_norm",
                       choices=["batch_norm"],
                       required=False, help="Normalization method (default: %(default)s)")

    group = parser.add_argument_group(title="Training Arguments")
    group.add_argument("--weight_decay_rate",
                       type=float,
                       default=1e-5,
                       required=False, help="Weight decay rate for variable regularizers (default: %(default)f)")
    group.add_argument("--bias_decay",
                       action="store_true",
                       required=False, help="Use bias decay or not")
    group.add_argument("--loss_weight_type",
                       type=str,
                       default="none",
                       choices=["none", "numerical", "proportion"],
                       required=False, help="Weights used in loss function for alleviating class imbalance problem"
                                            "(default %(default)s)")
    group.add_argument("--loss_numeric_w",
                       type=float,
                       nargs="+",
                       required=False, help="Numeric weights for loss_weight_type=\"numerical\". Notice that one value"
                                            "for one class")
    group.add_argument("--loss_proportion_decay",
                       type=float,
                       default=1000,
                       required=False, help="Proportion decay for loss_weight_type=\"proportion\". Check source code"
                                            "for details. (default: %(default)f)")
    group.add_argument("--metrics_train",
                       type=str,
                       default="Dice",
                       choices=["Dice", "VOE", "VD"],
                       nargs="+",
                       required=False, help="Evaluation metric names (default: %(default)s)")

    group = parser.add_argument_group(title="Evaluation Arguments")
    group.add_argument("--metrics_eval",
                       type=str,
                       default="Dice",
                       choices=["Dice", "VOE", "RVD", "ASSD", "RMSD", "MSD"],
                       nargs="+",
                       required=False, help="Evaluation metric names (default: %(default)s)")


def check_args(args, parser):
    if args.zoom_scale < 1:
        raise parser.error("Asserting {} >= 1 failed!".format(args.zoom_scale))
    if len(args.loss_numeric_w) != len(args.classes) + 1:
        raise parser.error("Asserting len(args.loss_numeric_w) = len(args.classes) + 1 failed!")


def main():
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args()
    check_args(args, parser)
    print(type(args))


if __name__ == "__main__":
    main()
