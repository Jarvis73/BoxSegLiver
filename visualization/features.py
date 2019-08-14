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
import pandas as pd
from pandas.plotting import scatter_matrix
from scipy.stats import gaussian_kde


def factor_scatter_matrix(df, factor, palette=None):
    """Create a scatter matrix of the variables in df, with differently colored
    points depending on the value of df[factor].
    inputs:
        df: pandas.DataFrame containing the columns to be plotted, as well
            as factor.
        factor: string or list or pandas.Series. The column indicating which group
            each row belongs to.
        palette: A list of hex codes, at least as long as the number of groups.
            If omitted, a predefined palette will be used, but it only includes
            9 groups.
    """
    if isinstance(factor, str):
        factor_name = factor    # save off the name
        factor = df[factor]     # extract column
        df = df.drop(factor_name, axis=1)    # remove from df, so it
        # doesn't get a row and col in the plot.
    if isinstance(factor, (tuple, list)):
        factor = pd.Series(factor)

    classes = list(set(factor))

    if palette is None:
        palette = ['#e41a1c', '#377eb8', '#4eae4b',
                   '#fdfc33', '#ff8101', '#994fa1',
                   '#a8572c', '#f482be', '#999999']

    color_map = dict(zip(classes, palette))

    if len(classes) > len(palette):
        raise ValueError("Too many groups for the number of colors provided."
                         "We only have {} colors in the palette, but you have {}"
                         "groups.".format(len(palette), len(classes)))

    colors = factor.apply(lambda group: color_map[group])
    axarr = scatter_matrix(df, alpha=1.0, figsize=(10, 10), marker='o', c=colors, diagonal=None)

    for rc in range(len(df.columns)):
        for group in classes:
            y = df[factor == group].iloc[:, rc].values
            assert y.shape[0] > 0, y.shape
            gkde = gaussian_kde(y)
            ind = np.linspace(y.min(), y.max(), 1000)
            _ = axarr[rc][rc].plot(ind, gkde.evaluate(ind), c=color_map[group])

    return axarr, color_map
