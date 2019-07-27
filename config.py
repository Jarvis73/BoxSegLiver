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


class CustomKeys(object):
    LEARNING_RATE = "learning_rate"
    LOSS_MEAN = "total_loss_mean"
    LR_UPDATE_OPS = "lr_update_ops"


def add_arguments(parser):
    group = parser.add_argument_group(title="Global Arguments")
    group.add_argument("--mode",
                       type=str,
                       choices=["train", "eval", "infer"],
                       required=True, help="Model mode for train/val/test")
    group.add_argument("--tag",
                       type=str,
                       required=True, help="Configuration tag(like UID)")
    group.add_argument("--model_dir",
                       type=str,
                       default="", help="Directory to save model parameters, graph and etc")
    group.add_argument("-s", "--save_predict",
                       action="store_true", help="Save prediction to file")
    group.add_argument("--warm_start_from",
                       type=str, help="Warm start the model from a checkpoint")
    group.add_argument("-l", "--load_status_file",
                       type=str,
                       default="checkpoint", help="Status file to locate checkpoint file. Use for restore"
                                                  "parameters.")
    group.add_argument("--out_file",
                       type=str, help="Logging file name to replace default.")
    group.add_argument("--summary_prefix", type=str,
                       help="A string that will be prepend to the summary tags. It is useful for "
                            "differentiating experiments with different prefix or merge several "
                            "experiments into a single label with the same prefix. Default is set"
                            "to experiment tag.")

    group = parser.add_argument_group(title="Device Arguments")
    group.add_argument("--distribution_strategy",
                       type=str,
                       default="off",
                       choices=['off', 'default', 'one_device', 'mirrored', 'parameter_server'],
                       required=False, help="A string specify which distribution strategy to use "
                                            "(default: %(default)s)")
    group.add_argument("--num_gpus",
                       type=int,
                       default=1,
                       required=False, help="Number of gpus to run this model")
    group.add_argument("--all_reduce_alg",
                       type=str,
                       default="",
                       choices=["", "hierarchical_copy", "nccl"],
                       required=False, help="Specify which algorithm to use when performing all-reduce")
    group.add_argument("--device_mem_frac",
                       type=float,
                       default=0.,
                       required=False, help="Used for per_process_gpu_memory_fraction")


def _try_to_find_ckpt(path, args):
    path = Path(path)

    if path.exists() or path.with_suffix(path.suffix + ".index").exists():   # Absolute path
        return str(path)
    cur_path = Path(__file__).parent / path
    if cur_path.exists() or cur_path.with_suffix(path.suffix + ".index").exists():   # Relative path
        return str(cur_path)
    model_dir = "model_dir" if not args.model_dir else args.model_dir
    cur_path = Path(__file__).parent / model_dir / path
    if cur_path.exists() or cur_path.with_suffix(path.suffix + ".index").exists():   # Relative path
        return str(cur_path)
    raise FileNotFoundError(path)


def check_args(args, parser):
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

    if args.warm_start_from:
        args.warm_start_from = _try_to_find_ckpt(args.warm_start_from, args)

    if not args.summary_prefix:
        args.summary_prefix = args.tag


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
