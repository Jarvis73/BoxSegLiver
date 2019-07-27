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

import json
import shutil
from pathlib import Path

import numpy as np


def random_split_k_fold(list_, k, seed=None):
    state = np.random.get_state()
    np.random.seed(seed)

    np.random.shuffle(list_)
    num_total = len(list_)
    num_test = int(num_total / k)

    folds = []
    for i in range(k):
        folds.append(list_[i * num_test:(i + 1) * num_test])

    remain = list_[k * num_test:]
    for i, f in enumerate(remain):
        folds[i].append(f)

    np.random.set_state(state)

    return folds


def read_or_create_k_folds(path, list_, k_split=None, seed=None):
    path = Path(path)

    if path.exists():
        k_folds = []
        with path.open() as f:
            for line in f.readlines():
                k_folds.append(line[line.find(":") + 1:].strip().split(" "))
    else:
        if not isinstance(k_split, int) or k_split <= 0:
            raise ValueError("Wrong `k_split` value. Need a positive integer, got {}".format(k_split))
        if k_split < 1:
            raise ValueError("k_split should >= 1, but get {}".format(k_split))
        k_folds = random_split_k_fold(list_, k_split, seed) if k_split > 1 else [list_]

        with path.open("w") as f:
            for i, fold in enumerate(k_folds):
                f.write("Fold %d:" % i)
                write_str = " ".join([str(x) for x in fold])
                f.write(write_str + "\n")

    try:
        # sort integers
        for fold in k_folds:
            print(",".join([str(x) for x in sorted([int(x) for x in fold])]))
    except ValueError:
        for fold in k_folds:
            print(",".join(fold))
    return k_folds


def load_meta(dataset, find_path):
    """
    Try to load meta.json

    Parameters
    ----------
    dataset: str
        The same with folder name in `<PROJECT>/DataLoader`
    find_path: str
        Path after `<PROJECT>/data` to find meta.json

    For example:
        load_meta("Liver", "LiTS/png")

    Returns
    -------
    meta dict
    """
    prepare_dir = Path(__file__).parent / dataset / "prepare"
    # Load meta.json
    meta_file = prepare_dir / "meta.json"
    if not meta_file.exists():
        src_meta = Path(__file__).parent.parent / "data" / find_path / "meta.json"
        if not src_meta.exists():
            raise FileNotFoundError(str(src_meta))
        shutil.copyfile(str(src_meta), str(meta_file))
    with meta_file.open() as f:
        meta = json.load(f)
    return meta
