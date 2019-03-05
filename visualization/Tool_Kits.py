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

import re
from pathlib import Path


def get_pred_score(log_file, sort_by=None):
    """

    Parameters
    ----------
    log_file: str or Path
        path to log file
    sort_by: str
        sort results by 'name' or 'score', or None

    Returns
    -------

    """
    data = Path(log_file).read_text()
    pat = re.compile("Evaluate-\d+\s(.*?)\s.*?/Dice:\s(\d+\.\d{3})\s.*?/Dice:\s(\d+\.\d{3})")

    res = pat.findall(data)
    res = [(x, (float(y), float(z))) for x, y, z in res]

    if sort_by is None:
        return res
    elif sort_by == "name":
        ind = 0
    else:
        raise ValueError("Only support sort results by 'name' or 'score'.")

    return sorted(res, key=lambda x: float(x[ind]))
