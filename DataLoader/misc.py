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
        for fold in k_folds:
            print(sorted([int(x) for x in fold]))
    except ValueError:
        for fold in k_folds:
            print(fold)
    return k_folds

