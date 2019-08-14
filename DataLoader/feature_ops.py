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


def hist_preprocess(feat, **kwargs):
    if "hist_scale" in kwargs:
        feat *= kwargs["hist_scale"]
    return feat.astype(np.float32)


def glcm_preprocess(feat, **kwargs):
    _ = kwargs
    return feat.astype(np.float32)
