import cv2
import sys
import numpy as np
from pathlib import Path
import qimage2ndarray as q2a

from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox, QFrame
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
        self.settings = QSettings('Jarvis', 'NetworkExplorerNF')
        self.init_data()
        self.init_action()
        self.init_state()
        self.init_window()

    def init_window(self):
        # Initial window size/pos last saved. Use default values for first time
        self.resize(self.settings.value("size", QSize(270, 225)))
        self.move(self.settings.value("pos", QPoint(100, 100)))
        self.splitter.restoreState(self.settings.value("splitterState", self.splitter.saveState()))
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
        for i in range(3):
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
        self.last_label = None

    def display_page(self, page):
        self.page.setText(str(page))
        first_idx_this_page = 24 * (page - 1)
        for i in range(3):
            for j in range(8):
                if first_idx_this_page + i * 8 + j >= self.total:
                    qImg = q2a.gray2qimage(np.zeros((self.thumb_height, self.thumb_width), np.float32), normalize=True)
                else:
                    qImg = q2a.gray2qimage(self.thumbs[..., first_idx_this_page + i * 8 + j], normalize=True)
                label = eval("self.label_%d%d" % (i, j))
                label.setPixmap(QPixmap.fromImage(qImg))
                label.sid = first_idx_this_page + i * 8 + j

    def labelButton_clicked(self, event):
        if self.feature_maps is None:
            return
        if 0 <= self.sender().sid < self.total:
            if event.button() == Qt.LeftButton:
                self.show_sid = self.sender().sid
                qImg = q2a.gray2qimage(self.feature_maps[..., self.show_sid], normalize=True)
                self.label.setPixmap(QPixmap.fromImage(qImg))
                # Change apperance of current thumb for clarity
                idx = self.sender().sid % 24
                cur_label = eval("self.label_{}{}".format(idx // 8, idx % 8))
                if cur_label != self.last_label:
                    cur_label.setStyleSheet("""
                                        QLabel {
                                            border: 2px solid blue;
                                            border-radius: 4px;
                                            padding: 2px;
                                        }
                                    """)
                    if self.last_label:
                        self.last_label.setStyleSheet("")
                self.last_label = cur_label
            elif event.button() == Qt.RightButton:
                sid = self.sender().sid
                save_path = Path(self.npy_dir).parent / "{}-{}.png".format(self.selected_npy.stem, sid)
                save_name = QFileDialog.getSaveFileName(self, 'Save File', str(save_path),
                                                        filter="Images (*.png *.jpg)")[0]
                print(save_name)
                if save_name:
                    save_img = self.feature_maps[..., sid]
                    min_ = save_img.min()
                    save_img = (save_img - min_) * (255. / (save_img.max() - min_))
                    status = cv2.imwrite(save_name, save_img.astype(np.uint8))
                    if not status:
                        QMessageBox.critical(self, "Error", "Save failed!", QMessageBox.Ok)
                    else:
                        print("Successfully saved {}".format(save_name))

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
        self.selected_npy = self.npy_files[qModelIndex.row()]
        self.feature_maps = np.load(str(self.selected_npy))[0]
        self.height, self.width = self.feature_maps.shape[:-1]
        self.thumbs = self.feature_maps[::4, ::4]
        self.thumb_height, self.thumb_width, self.total = self.thumbs.shape
        self.pageTotal.setText(str((self.total + 23) // 24))
        self.display_page(1)        

    def lastPageButton_clicked(self):
        curPage = int(self.page.text())
        if curPage > 1:
            self.display_page(curPage - 1)
            if self.last_label:
                self.last_label.setStyleSheet("")
                self.last_label = None
    
    def nextPageButton_clicked(self):
        curPage = int(self.page.text())
        if curPage < int(self.pageTotal.text()):
            self.display_page(curPage + 1)
            if self.last_label:
                self.last_label.setStyleSheet("")
                self.last_label = None

    def firstPageButton_clicked(self):
        curPage = int(self.page.text())
        if curPage != 1:
            self.display_page(1)
            if self.last_label:
                self.last_label.setStyleSheet("")
                self.last_label = None

    def finalPageButton_clicked(self):
        curPage = int(self.page.text())
        if curPage != int(self.pageTotal.text()):
            self.display_page(int(self.pageTotal.text()))
            if self.last_label:
                self.last_label.setStyleSheet("")
                self.last_label = None

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
