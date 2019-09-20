import sys
import numpy as np
from pathlib import Path
import qimage2ndarray as q2a

from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog
from PyQt5.QtCore import QStringListModel, QCoreApplication, Qt, QSettings, QPoint, QSize
from PyQt5.QtGui import QIcon, QImage, QPixmap

from ui import *

proj_root = Path(__file__).absolute().parent.parent.parent
viewer_root = Path(__file__).absolute().parent


# class ClickQLabel(QtWidgets.QLabel):
#     clicked = QtCore.pyqtSignal()
#     sid = -1

#     def mouseReleaseEvent(self, event):
#         self.clicked.emit()


# class MoveQLabel(QtWidgets.QLabel):
#     moved = QtCore.pyqtSignal()

#     def mouseMoveEvent(self, event):
#         self.moved.emit()

# self.label_(\d\d) = QtWidgets.QLabel\(self.verticalLayoutWidget_3\)
# self.label_$1 = EventQLabel(self.verticalLayoutWidget_3)


class mainWin(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(mainWin, self).__init__(parent)
        self.setupUi(self)
        self.setWindowTitle("Network Explorer")
        self.setWindowIcon(QIcon(str(viewer_root / "Assets/icon.png")))
        self.settings = QSettings('Jarvis', 'NetworkExplorer')
        self.init_data()
        self.init_action()
        self.init_state()
        self.init_window()

    def init_window(self):
        # Initial window size/pos last saved. Use default values for first time
        self.resize(self.settings.value("size", QSize(270, 225)))
        self.move(self.settings.value("pos", QPoint(100, 100)))
        self.splitter.restoreState(self.settings.value("splitterState"))
        self.npy_dir = self.settings.value("history_model_path", "")

    def init_state(self):
        self.page.setText("0")
        self.pageTotal.setText("0")

    def init_action(self):
        self.loadModelButton.clicked.connect(self.loadModelButton_clicked)
        self.listView.clicked.connect(self.listView_clicked)
        self.lastPageButton.clicked.connect(self.lastPageButton_clicked)
        self.nextPageButton.clicked.connect(self.nextPageButton_clicked)
        self.firstPageButton.clicked.connect(self.firstPageButton_clicked)
        self.finalPageButton.clicked.connect(self.finalPageButton_clicked)
        for i in range(8):
            for j in range(8):
                label = eval("self.label_%d%d" % (i, j))
                label.clicked.connect(self.labelButton_clicked)
        self.label.moved.connect(self.label_moved)
        self.label.setMouseTracking(True)

    def init_data(self):
        self.npy_dir = []
        self.feature_maps = None
        self.width = 0
        self.height = 0
        self.thumb_width = 0
        self.thumb_height = 0
        self.total = 0
        self.show_sid = -1

    def display_page(self, page):
        self.page.setText(str(page))
        first_idx_this_page = 64 * (page - 1)
        for i in range(8):
            for j in range(8):
                if first_idx_this_page + i * 8 + j >= self.total:
                    qImg = q2a.gray2qimage(np.zeros((self.thumb_height, self.thumb_width), np.float32), normalize=True)
                else:
                    qImg = q2a.gray2qimage(self.thumbs[..., first_idx_this_page + i * 8 + j], normalize=True)
                label = eval("self.label_%d%d" % (i, j))
                label.setPixmap(QPixmap.fromImage(qImg))
                label.sid = first_idx_this_page + i * 8 + j

    def labelButton_clicked(self):
        if self.feature_maps is None:
            return
        self.show_sid = self.sender().sid
        qImg = q2a.gray2qimage(self.feature_maps[..., self.show_sid], normalize=True)
        self.label.setPixmap(QPixmap.fromImage(qImg))

    def loadModelButton_clicked(self):
        _translate = QCoreApplication.translate
        self.npy_dir = str(QFileDialog.getExistingDirectory(self, "Select Directory", self.npy_dir))
        self.modelPath.setText(_translate("MainWindow", self.npy_dir))
        self.npy_files = list(Path(self.npy_dir).glob("*.npy")) 
        self.slm = QStringListModel()
        self.slm.setStringList([str(x.name) for x in self.npy_files])
        self.listView.setModel(self.slm)
        self.settings.setValue("history_model_path", str(Path(self.npy_dir).parent))

    def listView_clicked(self, qModelIndex):
        selected_npy = self.npy_files[qModelIndex.row()]
        self.feature_maps = np.load(str(selected_npy))[0]
        self.height, self.width = self.feature_maps.shape[:-1]
        self.thumbs = self.feature_maps[::4, ::4]
        self.thumb_height, self.thumb_width, self.total = self.thumbs.shape
        self.pageTotal.setText(str((self.total + 63) // 64))
        self.display_page(1)        

    def lastPageButton_clicked(self):
        curPage = int(self.page.text())
        if curPage > 1:
            self.display_page(curPage - 1)
    
    def nextPageButton_clicked(self):
        curPage = int(self.page.text())
        if curPage < int(self.pageTotal.text()):
            self.display_page(curPage + 1)

    def firstPageButton_clicked(self):
        curPage = int(self.page.text())
        if curPage != 1:
            self.display_page(1)

    def finalPageButton_clicked(self):
        curPage = int(self.page.text())
        if curPage != int(self.pageTotal.text()):
            self.display_page(int(self.pageTotal.text()))

    def label_moved(self, event):
        size = self.label.size()
        height, width = size.height(), size.width()
        x, y = event.x(), event.y()
        if self.show_sid >= 0:
            i = int(y / (height - 1) * (self.height - 1))
            j = int(x / (width - 1) * (self.width - 1))
            v = self.feature_maps[i, j, self.show_sid]
        else:
            i = 0
            j = 0
            v = 0
        self.position.setText("X: %3d Y: %3d | I: %3d J: %3d | V: %.3f" % (x, y, i, j, v))

    def closeEvent(self, e):
        # Write window size and position to config file
        self.settings.setValue("size", self.size())
        self.settings.setValue("pos", self.pos())
        self.settings.setValue("splitterState", self.splitter.saveState())
        e.accept()
        super(mainWin, self).closeEvent(e)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_win = mainWin()
    main_win.show()
    sys.exit(app.exec_())
