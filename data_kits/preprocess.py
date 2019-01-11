# Copyright 2019 Zhang Jianwei All Right Reserved.
#
# TODO: Choice a License
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
