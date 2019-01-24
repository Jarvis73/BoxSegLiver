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
import numpy as np


def random_split_k_fold(file_list, k, seed=None):
    state = np.random.get_state()
    np.random.seed(seed)

    np.random.shuffle(file_list)
    num_total = len(file_list)
    num_test = int(num_total / k)

    folds = []
    for i in range(k):
        folds.append(file_list[i * num_test:(i + 1) * num_test])

    remain = file_list[k * num_test:]
    for i, f in enumerate(remain):
        folds[i].append(f)

    np.random.set_state(state)

    return folds
