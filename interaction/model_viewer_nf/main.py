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
        self.npy_dir = ""
        self.cmp_npy_dir = ""
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
        self.cmp_npy_dir = self.settings.value("history_cmp_model_path", "")

    def init_state(self):
        self.page.setText("0")
        self.pageTotal.setText("0")

    def init_action(self):
        self.loadModelButton.clicked.connect(self.loadModelButton_clicked)
        self.loadCmpModelButton.clicked.connect(self.loadCmpModelButton_clicked)
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
        self.label.clicked.connect(self.bigLabelButton_clicked)
        self.label.setMouseTracking(True)
        self.ori_image.moved.connect(self.ori_image_moved)
        self.ori_image.clicked.connect(self.bigLabelButton_clicked)
        self.ori_image.setMouseTracking(True)
        self.cmp_image.moved.connect(self.cmp_image_moved)
        self.cmp_image.clicked.connect(self.bigLabelButton_clicked)
        self.cmp_image.setMouseTracking(True)

    def init_data(self):
        self.show_sid = -1
        self.last_label = None
        self.ori_file = None
        self.ori_slice = None
        self.ori_height = 0
        self.ori_width = 0
        self.isCmp = False
        self.reset_npy_file()

    def reset_npy_file(self):
        self.width = 0
        self.height = 0
        self.thumb_width = 0
        self.thumb_height = 0
        self.total = 0
        self.feature_maps = None
        self.cmp_feature_maps = None

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

    def change_sid(self, sid):
        self.show_sid = sid
        qImg = q2a.gray2qimage(self.feature_maps[..., self.show_sid], normalize=True)
        self.label.setPixmap(QPixmap.fromImage(qImg))
        # If cmp exists
        if self.isCmp and self.cmp_feature_maps is not None:
            qImg = q2a.gray2qimage(self.cmp_feature_maps[..., self.show_sid],
                                   normalize=True)
            self.cmp_image.setPixmap(QPixmap.fromImage(qImg))
        # Change apperance of current thumb for clarity
        idx = sid % 24
        cur_label = eval("self.label_{}{}".format(idx // 8, idx % 8))
        cur_label.setFocus()
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

    def save_img(self, save_path, sid=None, save_which=None):
        save_name = QFileDialog.getSaveFileName(self, 'Save File', str(save_path),
                                                filter="Images (*.png *.jpg)")[0]
        if save_name:
            if sid is not None:
                if save_which == "cmp":
                    save_img = self.cmp_feature_maps[..., sid]
                else:
                    save_img = self.feature_maps[..., sid]
            else:   # save image
                save_img = self.ori_slice
            min_ = save_img.min()
            save_img = (save_img - min_) * (255. / (save_img.max() - min_))
            status = cv2.imwrite(save_name, save_img.astype(np.uint8))
            if not status:
                QMessageBox.critical(self, "Error", "Save failed!", QMessageBox.Ok)
            else:
                print("Successfully saved {}".format(save_name))

    def labelButton_clicked(self, event):
        if self.feature_maps is None:
            return
        if 0 <= self.sender().sid < self.total:
            if event.button() == Qt.LeftButton:
                self.change_sid(self.sender().sid)
            elif event.button() == Qt.RightButton:
                sid = self.sender().sid
                save_path = Path(self.npy_dir).parent / "{}-{}.png".format(self.selected_npy.stem, sid)
                self.save_img(save_path, sid)

    def bigLabelButton_clicked(self, event):
        if event.button() == Qt.RightButton:
            if self.sender().objectName() == "ori_image" and self.ori_file is not None:
                save_path = Path(self.npy_dir).parent / self.ori_file.with_suffix(".png").name
                self.save_img(save_path)
            elif self.feature_maps is None:
                return
            if self.sender().objectName() == "label" and 0 <= self.show_sid < self.total:
                save_path = Path(self.npy_dir).parent / "{}-{}.png".format(self.selected_npy.stem, self.show_sid)
                self.save_img(save_path, self.show_sid)
            elif self.sender().objectName() == "cmp_image" and 0 <= self.show_sid < self.total:
                save_path = Path(self.cmp_npy_dir).parent / "{}-{}.png".format(self.cmp_selected_npy.stem,
                                                                               self.show_sid)
                self.save_img(save_path, self.show_sid, save_which="cmp")

    def loadModelButton_clicked(self):
        self.init_data()
        _translate = QCoreApplication.translate
        self.npy_dir = str(QFileDialog.getExistingDirectory(self, "Select Directory", self.npy_dir))
        self.modelPath.setText(_translate("MainWindow", self.npy_dir))
        self.npy_files = list(Path(self.npy_dir).glob("*.npy")) 
        self.slm = QStringListModel()
        self.slm.setStringList([str(x.name) for x in self.npy_files])
        self.listView.setModel(self.slm)
        ori_name = self.parse_ori_image(Path(self.npy_dir).name)
        ori_dir = Path(self.npy_dir).parent / "inputs_npy"
        if ori_name is not None and ori_dir.exists():
            self.ori_file = Path(ori_dir / ori_name)
        self.load_ori()
        self.isCmp = False

    def loadCmpModelButton_clicked(self):
        _translate = QCoreApplication.translate
        self.cmp_npy_dir = str(QFileDialog.getExistingDirectory(self, "Select Directory", self.cmp_npy_dir))
        self.cmpModelPath.setText(_translate("MainWindow", self.cmp_npy_dir))
        self.cmp_npy_files = list(Path(self.cmp_npy_dir).glob("*.npy"))
        self.isCmp = True

    def listView_clicked(self, qModelIndex):
        self.reset_npy_file()
        self.selected_npy = self.npy_files[qModelIndex.row()]
        self.feature_maps = np.load(str(self.selected_npy))[0]
        self.height, self.width = self.feature_maps.shape[:-1]
        self.thumbs = self.feature_maps[::4, ::4]
        self.thumb_height, self.thumb_width, self.total = self.thumbs.shape
        self.pageTotal.setText(str((self.total + 23) // 24))
        if self.last_label:
            self.last_label.setStyleSheet("")
            self.last_label = None
        self.display_page(1)
        # If we have cmp dir
        if self.isCmp:
            self.cmp_selected_npy = self.cmp_npy_files[qModelIndex.row()]
            self.cmp_feature_maps = np.load(str(self.cmp_selected_npy))[0]

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
        self.position.setText("X: %3d Y: %3d \nI: %3d J: %3d \nV: %.3f" % (x, y, i, j, v))

    def ori_image_moved(self, event):
        size = self.ori_image.size()
        height, width = size.height(), size.width()
        x, y = event.x(), event.y()
        if self.ori_slice is not None:
            i = int(y / (height - 1) * (self.ori_height - 1))
            j = int(x / (width - 1) * (self.ori_width - 1))
            v = self.ori_slice[i, j] * 900
        else:
            i = 0
            j = 0
            v = 0
        self.position2.setText("X: %3d Y: %3d \nI: %3d J: %3d \nV: %.3f" % (x, y, i, j, v))

    def cmp_image_moved(self, event):
        size = self.cmp_image.size()
        height, width = size.height(), size.width()
        x, y = event.x(), event.y()
        if self.show_sid >= 0 and self.cmp_feature_maps is not None:
            i = int(y / (height - 1) * (self.height - 1))
            j = int(x / (width - 1) * (self.width - 1))
            v = self.cmp_feature_maps[i, j, self.show_sid]
        else:
            i = 0
            j = 0
            v = 0
        self.position3.setText("X: %3d Y: %3d \nI: %3d J: %3d \nV: %.3f" % (x, y, i, j, v))

    def load_ori(self):
        if self.ori_file is None:
            return
        self.ori_slice = np.load(self.ori_file)[0, :, :, 1]
        qImg = q2a.gray2qimage(self.ori_slice, normalize=True)
        self.ori_image.setPixmap(QPixmap.fromImage(qImg))
        self.ori_height, self.ori_width = self.ori_slice.shape

    def parse_ori_image(self, name):
        parts = str(name).split("_")
        if len(parts) > 2:
            parts = parts[2:]
            if parts[-1] == "g":
                parts = parts[:-1]
            ori_name = "-".join(parts) + ".npy"
        else:
            ori_name = None
        return ori_name

    def closeEvent(self, e):
        # Write window size and position to config file
        self.settings.setValue("size", self.size())
        self.settings.setValue("pos", self.pos())
        self.settings.setValue("splitterState", self.splitter.saveState())
        if self.npy_dir:
            self.settings.setValue("history_model_path", str(Path(self.npy_dir).parent))
        if self.cmp_npy_dir:
            if self.isCmp:
                self.settings.setValue("history_cmp_model_path", str(Path(self.cmp_npy_dir).parent))
            else:
                self.settings.setValue("history_cmp_model_path", str(Path(self.cmp_npy_dir)))
        e.accept()
        super(mainWin, self).closeEvent(e)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Right:
            if self.total > 0 and self.show_sid < self.total - 1:
                if self.show_sid % 24 == 23:
                    self.nextPageButton_clicked()
                self.change_sid(self.show_sid + 1)
        elif event.key() == Qt.Key_Left:
            if self.total > 0 and self.show_sid > 0:
                if self.show_sid % 24 == 0:
                    self.lastPageButton_clicked()
                self.change_sid(self.show_sid - 1)
        elif event.key() == Qt.Key_Down:
            if self.total > 0 and self.show_sid < self.total - 8:
                if self.show_sid % 24 > 15:
                    self.nextPageButton_clicked()
                self.change_sid(self.show_sid + 8)
        elif event.key() == Qt.Key_Up:
            if self.total > 0 and self.show_sid > 7:
                if self.show_sid % 24 < 8:
                    self.lastPageButton_clicked()
                self.change_sid(self.show_sid - 8)

        event.accept()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_win = mainWin()
    main_win.show()
    sys.exit(app.exec_())
