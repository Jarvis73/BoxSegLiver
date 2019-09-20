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

import collections
import matplotlib
from traitsui.basic_editor_factory import BasicEditorFactory
from traits.etsconfig.api import ETSConfig
ETSConfig.toolkit = "qt4"
from matplotlib.figure import Figure
from traits.api import (
    HasTraits, Instance, Button, Directory, Str, List, Int, Bool, Float,
    Color, File, Any, Enum
)
from traitsui.api import (
    View, Item, VSplit, HSplit, VGroup, HGroup, TableEditor,
    Handler
)
from traitsui.table_column import ObjectColumn
import numpy as np

if ETSConfig.toolkit == "qt4":
    matplotlib.use("Qt5Agg")
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as Toolbar
    from traitsui.qt4.editor import Editor
    from pyface.qt import QtGui

from pathlib import Path
from PyQt5.QtCore import Qt
from collections import Iterable
from matplotlib.widgets import EllipseSelector
from matplotlib.patches import Ellipse


class MyList(list):
    """ Reimplementing a list with new __repr__
    """
    def __init__(self, iterable):
        if not isinstance(iterable, Iterable):
            iterable = [iterable]
        self._obj = iterable
        super(MyList, self).__init__(iterable)

    def __repr__(self):
        str_list = [str(x) for x in self._obj]
        return " ".join(str_list)

    def __setitem__(self, key, value):
        self._obj[key] = value
        super(MyList, self).__setitem__(key, value)

    def __str__(self):
        return self.__repr__()

    def __format__(self, fmt_spec=""):
        return self.__repr__()


class _QtFigureEditor(Editor):
    scrollable = True

    def init(self, parent):
        self.control = self._create_canvas(parent)
        self.set_tooltip()

    def update_editor(self):
        pass

    def _create_canvas(self, parent):
        _ = parent
        panel = QtGui.QWidget()

        def mousemoved(event):
            if event.xdata is not None:
                x, y = event.xdata, event.ydata
                name = "Axes"
                arr = self.value.axes[0].images[0].get_array()
                if int(y) < arr.shape[0] and int(x) < arr.shape[1]:
                    data = MyList(arr[int(y), int(x)])
                else:
                    name = "Figure"
                    data = "0"
            else:
                x, y = event.x, event.y
                name = "Figure"
                data = "0"

            panel.info.setText("%s: %g, %g (%s)" % (name, x, y, data))

        panel.mousemoved = mousemoved
        vbox = QtGui.QVBoxLayout()
        panel.setLayout(vbox)

        mpl_control = FigureCanvas(self.value)
        vbox.addWidget(mpl_control)
        if hasattr(self.value, "canvas_events"):
            for event_name, callback in self.value.canvas_events:
                mpl_control.mpl_connect(event_name, callback)

        mpl_control.mpl_connect("motion_notify_event", mousemoved)

        if self.factory.toolbar:
            toolbar = Toolbar(mpl_control, panel)
            vbox.addWidget(toolbar)

        panel.info = QtGui.QLabel(panel)
        vbox.addWidget(panel.info)

        return panel


class Gesture(object):
    Axial = 1
    Coronal = 2
    Sagittal = 3


class MPLFigureEditor(BasicEditorFactory):
    if ETSConfig.toolkit == "qt4":
        klass = _QtFigureEditor

    toolbar = Bool(True)


class MedicalItem(HasTraits):
    name = Str
    slices = Str


class ViewHandler(Handler):

    def setattr(self, info, object_, name, value):
        Handler.setattr(self, info, object_, name, value)

    def object_showButton_changed(self, info):
        if info.initialized:
            info.ui.title = "Image Viewer - {}".format(info.object.cur_case.name)


def key_press_event_wrapper(cls):
    def key_press_event_inner(event):
        cls.key_press_event(event)
    return key_press_event_inner


class Framework(HasTraits):
    root_path = Directory(entries=10)
    data_list = List(MedicalItem)
    cur_case = Any
    splitter = Str
    space = Str(" " * 2)
    contour = Bool(False)
    label = Enum(0, 1, 2)
    alpha = Float(0.3)
    title1 = Str("Image")
    title2 = Str("Spatial Guide")
    color1 = Color("red")

    cur_ind = Int
    total_ind = Int

    figure1 = Instance(Figure, ())
    figure2 = Instance(Figure, ())
    showButton = Button("Show")
    lastButton = Button("Last")
    nextButton = Button("Next")
    saveButton = Button("SaveInteraction")
    view = View(
        HGroup(
            VGroup(
                VGroup(
                    Item("root_path", width=250),
                ),
                Item("data_list",
                     editor=TableEditor(
                         columns=[ObjectColumn(name="name", width=0.7),
                                  ObjectColumn(name="slices", width=0.3), ],
                         auto_size=True,
                         orientation="vertical",
                         row_factory=MedicalItem,
                         editable=False,
                         selected="cur_case"
                     ),
                     show_label=False
                     ),
                show_border=True
            ),
            VSplit(
                HSplit(
                    VGroup(
                        HGroup(
                            Item("title1", show_label=False, style="readonly"),
                            Item("space", show_label=False, style="readonly"),
                            Item("color1", label="Color")
                        ),
                        Item("figure1", editor=MPLFigureEditor(toolbar=True), show_label=False, height=768)
                    ),
                    VGroup(
                        HGroup(
                            Item("title2", show_label=False, style="readonly"),
                            Item("space", show_label=False, style="readonly"),
                        ),
                        Item("figure2", editor=MPLFigureEditor(toolbar=True), show_label=False)
                    )
                ),
                HGroup(
                    Item("space", show_label=False, style="readonly"),
                    Item("showButton", show_label=False),
                    Item("label", style="custom"),
                    Item("contour"),
                    Item("alpha"),
                    Item("cur_ind", label="Index"),
                    Item("splitter", label="/", style="readonly"),
                    Item("total_ind", show_label=False, style="readonly"),
                    Item("space", show_label=False, style="readonly"),
                ),
                HGroup(
                    Item("space", show_label=False, style="readonly"),
                    Item("lastButton", show_label=False),
                    Item("nextButton", show_label=False),
                    Item("saveButton", show_label=False),
                    Item("space", show_label=False, style="readonly"),
                )
            ),
        ),
        width=1036,
        height=868,
        title="Image Viewer",
        resizable=True,
        handler=ViewHandler()
    )

    def __init__(self, adapter, base_size, **kw):
        super(Framework, self).__init__(**kw)

        self.adap = adapter
        self.base_size = base_size

        self.cur_ind = 0
        self.total_ind = 0
        self.accelerate = 1
        self.cur_show = ""
        self.gesture = Gesture.Axial
        self.patches = collections.defaultdict(list)
        self.connected = False

    def connect(self):
        self.wrapper = key_press_event_wrapper(self)
        self.wrapper.ES = EllipseSelector(self.figure1.axes[0], self.on_select, drawtype='line', interactive=True,
                                          button=1,
                                          lineprops=dict(color='black', linestyle='-', linewidth=2, alpha=0.5))
        # Figure events
        self.figure1.canvas.mpl_connect("button_press_event", self.button_press_event)
        self.figure1.canvas.mpl_connect("button_release_event", self.button_release_event)
        self.figure1.canvas.mpl_connect("scroll_event", self.scroll_event)
        self.figure1.canvas.mpl_connect("key_press_event", self.wrapper)
        self.figure1.canvas.mpl_connect("key_release_event", self.key_release_event)
        self.figure1.canvas.setFocusPolicy(Qt.ClickFocus)
        self.figure2.canvas.mpl_connect("scroll_event", self.scroll_event)

    def on_select(self, eclick, erelease):
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        center = (x1 + x2) / 2, (y1 + y2) / 2
        axes_length = np.abs(x1 - x2), np.abs(y1 - y2)
        elli = Ellipse(center, *axes_length, alpha=0.5)
        self.patches[self.cur_ind].append(elli)
        print(x1, y1, x2, y2)
        self.adap.update_interaction(self.cur_ind, center, axes_length)
        self.refresh(ignore_fig1=True)

    def scroll_event(self, event):
        if event.button == "down":
            self._nextButton_fired()
        else:
            self._lastButton_fired()

    def button_press_event(self, event):
        if event.button == 3:
            print(event.xdata, event.ydata)
            self.adap.update_interaction(self.cur_ind, (event.xdata, event.ydata), (13.4898,) * 2)
            self.refresh(ignore_fig1=True)
        self.figure1.canvas.setFocus()

    def button_release_event(self, event):
        _ = event
        self.accelerate = 1

    def key_press_event(self, event):
        print(event.key)
        if event.key == "control":
            self.contour = not self.contour
            self._contour_changed()
        elif event.key in ['E', 'e']:
            if self.wrapper.ES.active:
                print('EllipseSelector deactivated.')
                self.wrapper.ES.set_active(False)
            else:
                print('EllipseSelector activated.')
                self.wrapper.ES.set_active(True)
            self.refresh()

    def key_release_event(self, event):
        if event.key == "down":
            self._nextButton_fired()
        elif event.key == "up":
            self._lastButton_fired()
        elif event.key == "right":
            self._nextButton_fired()
        elif event.key == "left":
            self._lastButton_fired()
        elif event.key == "1":
            if self.gesture != Gesture.Axial:
                self.gesture = Gesture.Axial
                self.reset_index()
                self.refresh()
        elif event.key == "2":
            if self.gesture != Gesture.Coronal:
                self.gesture = Gesture.Coronal
                self.reset_index()
                self.refresh()
        elif event.key == "3":
            if self.gesture != Gesture.Sagittal:
                self.gesture = Gesture.Sagittal
                self.reset_index()
                self.refresh()
        elif event.key == "u":
            if len(self.patches[self.cur_ind]) > 0:
                self.patches[self.cur_ind].pop()
                self.adap.pop_interaction(self.cur_ind)
                self.refresh(ignore_fig1=True)

    def reset_index(self):
        self.cur_ind = self.adap.get_min_idx(self.gesture)
        self.total_ind = self.adap.get_num_slices(self.gesture) - 1

    def patch_show(self):
        for p in self.figure1.axes[0].patches:
            p.remove()
        for p in self.patches[self.cur_ind]:
            self.figure1.axes[0].add_patch(p)

    def image_show(self, ind, ignore_fig1=False):
        if not ignore_fig1:
            self.axesImage1.set_data(
                self.adap.get_slice1(ind, self.color1.getRgb()[:-1],
                                     alpha=self.alpha,
                                     contour=self.contour,
                                     ges=self.gesture))
        self.axesImage2.set_data(self.adap.get_slice2(ind, ges=self.gesture))
        if not self.connected:
            self.connect()
            self.connected = True
        self.cur_ind = self.adap.real_ind(ind)

    def update_figure(self):
        if self.figure1.canvas is not None:
            self.figure1.canvas.draw_idle()
        if self.figure2.canvas is not None:
            self.figure2.canvas.draw_idle()

    def refresh(self, ind=None, ignore_fig1=False):
        ind = ind if ind is not None else self.cur_ind
        self.image_show(ind, ignore_fig1)
        self.patch_show()
        self.update_figure()

    def _root_path_default(self):
        return self.adap.get_root_path()

    def _data_list_default(self):
        if Path(self.root_path).exists():
            return [MedicalItem(name=x, slices=y) for x, y in self.adap.get_file_list()]
        else:
            return []

    def _root_path_changed(self):
        if Path(self.root_path).exists():
            self.adap.update_root_path(self.root_path)
            self.pred_list = [MedicalItem(name=x, slices=y)
                              for x, y in self.adap.get_file_list()]

    def _cur_ind_changed(self):
        if 0 <= self.cur_ind <= self.total_ind and self.total_ind > 0:
            self.refresh()

    def _alpha_changed(self):
        if 0.0 <= self.alpha <= 1.0 and self.total_ind > 0:
            self.refresh()

    def _color1_changed(self):
        if self.total_ind > 0:
            self.refresh()

    def _contour_changed(self):
        if self.total_ind > 0:
            self.refresh()

    def _label_changed(self):
        if self.total_ind > 0:
            self.adap.update_lab(label=self.label)
            self.refresh()

    def _figure1_default(self):
        figure = Figure(figsize=(12 * (self.base_size[1] / self.base_size[0]), 12))
        figure.add_axes([0.0, 0.0, 1.0, 1.0])
        figure.axes[0].axis("off")
        self.axesImage1 = figure.axes[0].imshow(np.zeros(self.base_size), cmap="gray")
        return figure

    def _figure2_default(self):
        figure = Figure(figsize=(12 * (self.base_size[1] / self.base_size[0]), 12))
        figure.add_axes([0.0, 0.0, 1.0, 1.0])
        figure.axes[0].axis("off")
        data = np.zeros(self.base_size)
        data[0, 0] = 1
        self.axesImage2 = figure.axes[0].imshow(data, cmap="gray")
        return figure

    def _lastButton_fired(self):
        if self.total_ind > 0:
            self.refresh(self.cur_ind - self.accelerate)

    def _nextButton_fired(self):
        if self.total_ind > 0:
            self.refresh(self.cur_ind + self.accelerate)

    def _showButton_fired(self):
        if self.cur_case and self.cur_show != self.cur_case:
            self.adap.update_case(self.cur_case.name)
            self.reset_index()
            self.refresh()
            self.cur_show = self.cur_case

    def _saveButton_fired(self):
        self.adap.save_interaction()
