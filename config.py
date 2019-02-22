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
                       choices=["train", "eval", "infer"],
                       required=True, help="Model mode for train/val/test")
    group.add_argument("--tag",
                       type=str,
                       required=True, help="Configuration tag(like UID)")
    group.add_argument("--train_without_eval",
                       action="store_true",
                       required=False, help="Evaluate model during training")
    group.add_argument("--model_dir",
                       type=str,
                       default="",
                       required=False, help="Directory to save model parameters, graph and etc")

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
                       required=False, help="Weights used in loss function for alleviating class imbalance problem "
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
                       default=["Dice"],
                       choices=["Dice", "VOE", "VD"],
                       nargs="+",
                       required=False, help="Evaluation metric names (default: %(default)s)")


def check_args(args, parser):

    if args.zoom_scale < 1:
        raise parser.error("Asserting {} >= 1 failed!".format(args.zoom_scale))

    if args.loss_weight_type == "numerical":
        if not args.loss_numeric_w:
            raise parser.error("loss_weight_type==numerical need parameter: --loss_numeric_w")
        if len(args.loss_numeric_w) != len(args.classes) + 1:
            raise parser.error("Asserting len(args.loss_numeric_w) = len(args.classes) + 1 failed!")
    elif args.loss_weight_type == "proportion":
        if not args.loss_proportion_decay:
            raise parser.error("loss_weight_type==proportion need parameter: --loss_proportion_decay")

    if args.primary_metric:
        parts = args.primary_metric.split("/")
        if len(parts) == 2:
            if parts[0] not in args.classes or parts[1] not in args.metrics_eval:
                raise ValueError("Wrong primary_metric: {}".format(args.primary_metric))
    if args.secondary_metric:
        parts = args.secondary_metric.split("/")
        if len(parts) == 2:
            if parts[0] not in args.classes or parts[1] not in args.metrics_eval:
                raise ValueError("Wrong secondary_metric: {}".format(args.secondary_metric))

    if args.mode == ModeKeys.TRAIN:
        if not args.dataset_for_train:
            raise parser.error("TRAIN mode need parameter: --dataset_for_train")
        for x in args.dataset_for_train:
            record = Path(__file__).parent / "data" / x
            if not record.exists():
                raise parser.error("File not found: " + str(record))

    if args.mode == ModeKeys.EVAL:
        if not args.dataset_for_eval:
            raise parser.error("EVAL mode need parameter: --dataset_for_eval")
        for x in args.dataset_for_eval:
            record = Path(__file__).parent / "data" / x
            if not record.exists():
                raise parser.error("File not found: " + str(record))


def fill_default_args(args):
    if not args.model_dir:
        model_dir = Path(__file__).parent / "model_dir"
        if not model_dir.exists():
            model_dir.mkdir(exist_ok=True)
        args.model_dir = str(model_dir / args.tag)


def main():
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args()
    check_args(args, parser)
    print(type(args))


if __name__ == "__main__":
    main()
