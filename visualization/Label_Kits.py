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

import sys
# noinspection PyUnresolvedReferences
import vtk
from pathlib import Path
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QKeySequence
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from vtkmodules.vtkRenderingCorePython import vtkRenderer, vtkImageActor
from vtkmodules.vtkInteractionStylePython import vtkInteractorStyleTrackballCamera
from vtkmodules.vtkIOImagePython import vtkNIFTIImageReader


ROOT_DIR = Path(__file__).parent.parent


class Ui_MainWindow(object):
    def setup_Ui(self, MainWindow):
        self.mw = MainWindow

        grid = QGridLayout()
        grid.setColumnMinimumWidth(1, 512)

        layout = QVBoxLayout()
        layout.addWidget(self.mw.widget)

        mid_part = QWidget(self.mw)
        mid_part.setLayout(layout)
        grid.addWidget(mid_part, 0, 1, 5, 1)

        window_width_level = QGroupBox("Window Width/Level")
        wwl_layout = QGridLayout()
        wwl_layout.addWidget(QLabel("Window width"), 0, 0)
        wwl_layout.addWidget(QLabel("Window level"), 1, 0)
        width_spin_box = QSpinBox()
        width_spin_box.setMaximum(4095)
        width_spin_box.setMinimum(0)
        width_spin_box.setSingleStep(1)
        width_spin_box.setValue(450)
        wwl_layout.addWidget(width_spin_box, 0, 1)
        level_spin_box = QSpinBox()
        level_spin_box.setMaximum(3071)
        level_spin_box.setMinimum(-1024)
        level_spin_box.setSingleStep(1)
        level_spin_box.setValue(25)
        wwl_layout.addWidget(level_spin_box, 1, 1)
        window_width_level.setLayout(wwl_layout)
        grid.addWidget(window_width_level, 0, 2)

        frame = QFrame()
        frame.setAutoFillBackground(True)
        frame.setLayout(grid)
        self.mw.setCentralWidget(frame)

    def enable_status_bar(self):
        self.statusBar = self.mw.statusBar()
        self.statusBar.showMessage("Ready!")

    def create_menu_bar(self):
        openAction = QAction("&Open", self.mw)
        openAction.setShortcut(QKeySequence.Open)
        openAction.setToolTip("Open file")
        openAction.setStatusTip("Open file")
        openAction.triggered.connect(self.mw.openFile)

        exitAction = QAction("&Exit", self.mw)
        exitAction.setShortcut(QKeySequence.Quit)
        exitAction.setToolTip("Exit application")
        exitAction.setStatusTip("Exit application")
        exitAction.triggered.connect(self.mw.exitApp)

        menuBar = self.mw.menuBar()
        fileMenu = menuBar.addMenu("&File")
        fileMenu.addAction(openAction)
        fileMenu.addAction(exitAction)

    def create_left_part(self):
        pass


class MainWindow(QMainWindow, QApplication):
    def __init__(self, parent=None):
        QMainWindow.__init__(self, parent)
        self.ui = Ui_MainWindow()

        self.ren = vtkRenderer()
        self.widget = QVTKRenderWindowInteractor()

        # Setup UI
        self.ui.setup_Ui(self)
        self.ui.enable_status_bar()
        self.ui.create_menu_bar()

        self.setWindowTitle("Medical Image ToolBox")

    def vtk_init(self, reader):
        propPicker = vtk.vtkPropPicker()
        propPicker.PickFromListOn()

        self.viewer = vtk.vtkImageViewer2()
        self.viewer.SetInputData(reader.GetOutput())
        propPicker.AddPickList(self.viewer.GetImageActor())

        coordTextProp = vtk.vtkTextProperty()
        coordTextProp.SetFontSize(20)
        coordTextProp.SetVerticalJustificationToBottom()
        coordTextProp.SetJustificationToLeft()
        coordTextMapper = vtk.vtkTextMapper()
        coordTextMapper.SetInput("Location: ( 0, 0, 0 )\nValue: ")
        coordTextMapper.SetTextProperty(coordTextProp)
        coordTextActor = vtk.vtkActor2D()
        coordTextActor.SetMapper(coordTextMapper)
        coordTextActor.SetPosition(15, 10)

        cornerAnnotation = vtk.vtkCornerAnnotation()
        cornerAnnotation.SetLinearFontScaleFactor(2)
        cornerAnnotation.SetNonlinearFontScaleFactor(1)
        cornerAnnotation.SetMaximumFontSize(20)
        cornerAnnotation.GetTextProperty().SetColor(1, 1, 1)
        viewer.GetRenderer().AddViewProp(cornerAnnotation)

        pointTextProp = vtk.vtkTextProperty()
        pointTextProp.SetFontSize(20)
        pointTextProp.SetVerticalJustificationToTop()
        pointTextProp.SetJustificationToRight()
        pointTextMapper = vtk.vtkTextMapper()
        pointTextMapper.SetInput("( 123, 456, 789 )")
        pointTextMapper.SetTextProperty(pointTextProp)
        pointTextActor = vtk.vtkActor2D()
        pointTextActor.SetMapper(pointTextMapper)
        pointTextActor.SetPosition(1285, 690)

        renderWindowInteractor = vtk.vtkRenderWindowInteractor()
        viewer.SetupInteractor(renderWindowInteractor)

        self.viewer.SetupInteractor(self.widget)
        self.viewer.SetRenderWindow(self.widget.GetRenderWindow())
        self.viewer.Render()

    def openFile(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Choice file", str(ROOT_DIR),
                                                   "NIFTI Files(*.nii)")
        if not file_path:
            self.ui.statusBar.showMessage("Open file canceled")
            return

        reader = vtkNIFTIImageReader()
        reader.SetFileName(file_path)
        reader.TimeAsVectorOn()
        reader.Update()
        self.si = 0

    @staticmethod
    def create_slice_actor(data, extent):
        plane = vtkImageActor()
        plane.GetMapper().SetInputConnection(data)
        plane.SetDisplayExtent(extent)
        plane.InterpolateOn()
        plane.ForceOpaqueOn()
        return plane

    def exitApp(self):
        self.exit(0)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    window.widget.Initialize()  # Need this line to actually show the render inside Qt
    sys.exit(app.exec_())
