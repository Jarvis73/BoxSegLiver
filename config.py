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
from pathlib import Path
from tensorflow.python.estimator.model_fn import ModeKeys


def add_arguments(parser):
    group = parser.add_argument_group(title="Global Arguments")
    group.add_argument("--mode",
                       type=str,
                       default="TRAIN",
                       choices=["TRAIN", "EVAL", "PREDICT"],
                       required=True, help="Model mode for train/val/test (default: %(default)s)")
    group.add_argument("--tag",
                       type=str,
                       required=True, help="Configuration tag(like UID)")
    group.add_argument("--model_dir",
                       type=str,
                       required=True, help="Directory to save model parameters, graph and etc")

    group = parser.add_argument_group(title="Loss Arguments")
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
    for x in args.dataset_for_train:
        record = Path(__file__).parent / "data" / x
        if not record.exists():
            raise parser.error("File not found: " + str(record))
    if args.mode == ModeKeys.EVAL:
        if not args.dataset_for_eval:
            raise parser.error("EVAL mode need `dataset_for_eval`")
        for x in args.dataset_for_eval:
            record = Path(__file__).parent / "data" / x
            if not record.exists():
                raise parser.error("File not found: " + str(record))


def main():
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args()
    check_args(args, parser)
    print(type(args))


if __name__ == "__main__":
    main()
