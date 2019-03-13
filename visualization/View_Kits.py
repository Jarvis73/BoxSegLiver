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

import matplotlib
from traitsui.basic_editor_factory import BasicEditorFactory
from traits.etsconfig.api import ETSConfig
ETSConfig.toolkit = "qt4"
from matplotlib.figure import Figure
from traits.api import (
    HasTraits, Instance, Button, Directory, Str, List, Int, Bool, Float,
    Color, File, Any
)
from traitsui.api import (
    View, Item, VSplit, HSplit, VGroup, HGroup, FileEditor, TableEditor,
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
                data = MyList(self.value.axes[0].images[0].get_array()[int(y), int(x)])
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
    liver_score = Float
    tumor_score = Float


class ViewHandler(Handler):

    def setattr(self, info, object, name, value):
        Handler.setattr(self, info, object, name, value)

    def object_showButton_changed(self, info):
        if info.initialized:
            info.ui.title = "Image Viewer - {}".format(info.object.cur_case.name)


class ComparePrediction(HasTraits):
    root_path = Directory(entries=10)
    score_path = File(entries=10)
    pred_list = List(MedicalItem)
    cur_case = Any
    splitter = Str
    space = Str(" " * 20)
    contour = Bool(True)
    mask = Bool(False)
    liver = Bool(False)
    alpha = Float(0.3)
    title1 = Str("Phase2")
    title2 = Str("Prediction")
    color1 = Color("red")
    color2 = Color("yellow")
    label_num = Int(2)

    cur_ind = Int
    total_ind = Int

    figure1 = Instance(Figure, ())
    figure2 = Instance(Figure, ())
    showButton = Button("Show")
    lastButton = Button("Last")
    nextButton = Button("Next")
    view = View(
        HGroup(
            VGroup(
                VGroup(
                    Item("root_path", width=250),
                    Item("score_path"),
                ),
                Item("pred_list",
                     editor=TableEditor(
                         columns=[ObjectColumn(name="name", width=0.3),
                                  ObjectColumn(name="slices", width=0.2),
                                  ObjectColumn(name="liver_score", width=0.1),
                                  ObjectColumn(name="tumor_score", width=0.1), ],
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
                        Item("figure1", editor=MPLFigureEditor(toolbar=True), show_label=False, height=542)
                    ),
                    VGroup(
                        HGroup(
                            Item("title2", show_label=False, style="readonly"),
                            Item("space", show_label=False, style="readonly"),
                            Item("color2", label="Color")
                        ),
                        Item("figure2", editor=MPLFigureEditor(toolbar=True), show_label=False)
                    )
                ),
                HGroup(
                    Item("space", show_label=False, style="readonly"),
                    Item("showButton", show_label=False),
                    Item("liver"),
                    Item("contour"),
                    Item("mask"),
                    Item("alpha"),
                    Item("label_num", label="Label"),
                    Item("cur_ind", label="Index"),
                    Item("splitter", label="/", style="readonly"),
                    Item("total_ind", show_label=False, style="readonly"),
                    Item("lastButton", show_label=False),
                    Item("nextButton", show_label=False),
                    Item("space", show_label=False, style="readonly"),
                )
            ),
        ),
        width=1324,
        height=580,
        title="Image Viewer",
        resizable=True,
        handler=ViewHandler()
    )

    def __init__(self, adapter, **kw):
        super(ComparePrediction, self).__init__(**kw)

        self.adap = adapter

        self.cur_ind = 0
        self.total_ind = 0
        self.accelerate = 1
        self.cur_show = ""
        self.gesture = Gesture.Axial

    def connect(self):
        # Figure events
        self.figure1.canvas.mpl_connect("button_press_event", self.button_press_event)
        self.figure1.canvas.mpl_connect("button_release_event", self.button_release_event)
        self.figure1.canvas.mpl_connect("scroll_event", self.scroll_event)
        self.figure1.canvas.mpl_connect("key_press_event", self.key_press_event)
        self.figure1.canvas.mpl_connect("key_release_event", self.key_release_event)
        self.figure2.canvas.mpl_connect("button_press_event", self.button_press_event)
        self.figure2.canvas.mpl_connect("button_release_event", self.button_release_event)
        self.figure2.canvas.mpl_connect("scroll_event", self.scroll_event)

        self.figure1.canvas.setFocusPolicy(Qt.ClickFocus)

    def scroll_event(self, event):
        if event.button == "down":
            self._nextButton_fired()
        else:
            self._lastButton_fired()

    def button_press_event(self, event):
        if event.button == 1:
            self.accelerate = 3
        elif event.button == 3:
            self.accelerate = 6
        self.figure1.canvas.setFocus()

    def button_release_event(self, event):
        _ = event
        self.accelerate = 1

    def key_press_event(self, event):
        if event.key == "control":
            self.contour = False
            self._contour_changed()

    def key_release_event(self, event):
        if event.key == "control":
            self.contour = True
            self._contour_changed()
        elif event.key == "shift":
            self.liver = not self.liver
            self._liver_changed()
        elif event.key == "down":
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

    def reset_index(self):
        self.cur_ind = self.adap.get_min_idx(self.gesture)
        self.total_ind = self.adap.get_num_slices(self.gesture) - 1

    def image_show(self, ind):
        self.axesImage1.set_data(
            self.adap.get_slice1(ind, self.color1.getRgb()[:-1],
                                 alpha=self.alpha,
                                 contour=self.contour,
                                 mask_lab=self.mask,
                                 ges=self.gesture))
        self.axesImage2.set_data(
            self.adap.get_slice2(ind, self.color2.getRgb()[:-1],
                                 alpha=self.alpha,
                                 contour=self.contour,
                                 mask_lab=self.mask,
                                 ges=self.gesture))
        self.connect()
        self.cur_ind = self.adap.real_ind(ind)
        self.update_figure()

    def update_figure(self):
        if self.figure1.canvas is not None:
            self.figure1.canvas.draw_idle()
        if self.figure2.canvas is not None:
            self.figure2.canvas.draw_idle()

    def refresh(self):
        self.image_show(self.cur_ind)

    def _root_path_default(self):
        return self.adap.get_root_path()

    def _score_path_default(self):
        return ""

    def _pred_list_default(self):
        if Path(self.root_path).exists():
            return [MedicalItem(name=x, slices=y) for x, y in self.adap.get_file_list()]
        else:
            return []

    def _root_path_changed(self):
        if Path(self.root_path).exists():
            self.adap.update_root_path(self.root_path)
            self.pred_list = [MedicalItem(name=x, slices=y)
                              for x, y in self.adap.get_file_list()]

    def _score_path_changed(self):
        if Path(self.score_path).exists():
            self.pred_list = [MedicalItem(name=x, slices=y, liver_score=z1, tumor_score=z2)
                              for x, y, z1, z2 in self.adap.get_pair_list(self.score_path)]
        else:
            print("Warning: {} not exists".format(self.score_path))

    def _cur_ind_changed(self):
        if 0 <= self.cur_ind <= self.total_ind and self.total_ind > 0:
            self.refresh()

    def _alpha_changed(self):
        if 0.0 <= self.alpha <= 1.0 and self.total_ind > 0:
            self.refresh()

    def _color1_changed(self):
        if self.total_ind > 0:
            self.refresh()

    def _color2_changed(self):
        if self.total_ind > 0:
            self.refresh()

    def _contour_changed(self):
        if self.total_ind > 0:
            self.refresh()

    def _mask_changed(self):
        if self.total_ind > 0:
            self.refresh()

    def _liver_changed(self):
        if self.total_ind > 0:
            self.adap.update_choice(liver=self.liver)
            self.refresh()

    def _label_num_changed(self):
        if self.total_ind > 0:
            self.adap.update_choice(label=self.label_num)
            self.refresh()

    def _figure1_default(self):
        figure = Figure()
        figure.add_axes([0.0, 0.0, 1.0, 1.0])
        figure.axes[0].axis("off")
        self.axesImage1 = figure.axes[0].imshow(np.ones((512, 512)), cmap="gray")
        return figure

    def _figure2_default(self):
        figure = Figure()
        figure.add_axes([0.0, 0.0, 1.0, 1.0])
        figure.axes[0].axis("off")
        self.axesImage2 = figure.axes[0].imshow(np.ones((512, 512)), cmap="gray")
        return figure

    def _lastButton_fired(self):
        if self.total_ind > 0:
            self.image_show(self.cur_ind - self.accelerate)

    def _nextButton_fired(self):
        if self.total_ind > 0:
            self.image_show(self.cur_ind + self.accelerate)

    def _showButton_fired(self):
        if self.cur_case and self.cur_show != self.cur_case:
            self.adap.update_case(self.cur_case.name, liver=self.liver,
                                  label=self.label_num)
            self.reset_index()
            self.refresh()
            self.cur_show = self.cur_case
