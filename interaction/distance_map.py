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
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from scipy.ndimage import morphology as mph


press = False
x = []
y = []
fig, ax = plt.subplots(1, 1)
a = np.zeros([256, 256], np.float32)
plt.imshow(a + 1, cmap="gray")
line, = plt.plot([], [])

def onclick(event):
    global press
    press = True

def onrelease(event):
    global press
    press = False

def onmove(event):
    if press:
        x.append(event.xdata)
        y.append(event.ydata)
        line.set_data(x, y)
        line.figure.canvas.draw()

fig.canvas.mpl_connect('button_press_event', onclick)
fig.canvas.mpl_connect('button_release_event', onrelease)
fig.canvas.mpl_connect('motion_notify_event', onmove)
plt.show()

disc = mph.generate_binary_structure(rank=2, connectivity=1)

xx, yy = line.get_data()
xx = np.array(xx, np.int32)
yy = np.array(yy, np.int32)
a[yy, xx] = 1

border = (a - mph.binary_erosion(a, disc)).astype(np.bool)
b = ndi.distance_transform_edt(~border)
c = np.exp(-b ** 2 / 25)
plt.imshow(c, cmap="gray")
plt.show()
