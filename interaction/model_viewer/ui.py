# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file '.\framework.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets


class ClickQLabel(QtWidgets.QLabel):
    clicked = QtCore.pyqtSignal(QtGui.QMouseEvent)
    sid = -1

    def mouseReleaseEvent(self, event):
        self.clicked.emit(event)


class MoveQLabel(QtWidgets.QLabel):
    moved = QtCore.pyqtSignal(QtGui.QMouseEvent)

    def mouseMoveEvent(self, event):
        self.moved.emit(event)


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1836, 941)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.splitter = QtWidgets.QSplitter(self.centralwidget)
        self.splitter.setOrientation(QtCore.Qt.Horizontal)
        self.splitter.setObjectName("splitter")
        self.verticalLayoutWidget_2 = QtWidgets.QWidget(self.splitter)
        self.verticalLayoutWidget_2.setObjectName("verticalLayoutWidget_2")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.verticalLayoutWidget_2)
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.modelPath = QtWidgets.QLabel(self.verticalLayoutWidget_2)
        self.modelPath.setMinimumSize(QtCore.QSize(300, 0))
        self.modelPath.setObjectName("modelPath")
        self.modelPath.setWordWrap(True)
        self.verticalLayout_2.addWidget(self.modelPath)
        self.loadModelButton = QtWidgets.QPushButton(self.verticalLayoutWidget_2)
        self.loadModelButton.setObjectName("loadModelButton")
        self.verticalLayout_2.addWidget(self.loadModelButton)
        self.listView = QtWidgets.QListView(self.verticalLayoutWidget_2)
        self.listView.setMinimumSize(QtCore.QSize(100, 0))
        self.listView.setMaximumSize(QtCore.QSize(400, 16777215))
        self.listView.setObjectName("listView")
        self.verticalLayout_2.addWidget(self.listView)
        self.verticalLayoutWidget_3 = QtWidgets.QWidget(self.splitter)
        self.verticalLayoutWidget_3.setObjectName("verticalLayoutWidget_3")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.verticalLayoutWidget_3)
        self.verticalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setSizeConstraint(QtWidgets.QLayout.SetDefaultConstraint)
        self.gridLayout.setContentsMargins(5, 0, 5, -1)
        self.gridLayout.setSpacing(5)
        self.gridLayout.setObjectName("gridLayout")
        self.label_24 = ClickQLabel(self.verticalLayoutWidget_3)
        self.label_24.setText("")
        self.label_24.setPixmap(QtGui.QPixmap(":/Assets/bg.png"))
        self.label_24.setScaledContents(True)
        self.label_24.setObjectName("label_24")
        self.gridLayout.addWidget(self.label_24, 2, 4, 1, 1)
        self.label_27 = ClickQLabel(self.verticalLayoutWidget_3)
        self.label_27.setText("")
        self.label_27.setPixmap(QtGui.QPixmap(":/Assets/bg.png"))
        self.label_27.setScaledContents(True)
        self.label_27.setObjectName("label_27")
        self.gridLayout.addWidget(self.label_27, 2, 7, 1, 1)
        self.label_10 = ClickQLabel(self.verticalLayoutWidget_3)
        self.label_10.setText("")
        self.label_10.setPixmap(QtGui.QPixmap(":/Assets/bg.png"))
        self.label_10.setScaledContents(True)
        self.label_10.setObjectName("label_10")
        self.gridLayout.addWidget(self.label_10, 1, 0, 1, 1)
        self.label_33 = ClickQLabel(self.verticalLayoutWidget_3)
        self.label_33.setText("")
        self.label_33.setPixmap(QtGui.QPixmap(":/Assets/bg.png"))
        self.label_33.setScaledContents(True)
        self.label_33.setObjectName("label_33")
        self.gridLayout.addWidget(self.label_33, 3, 3, 1, 1)
        self.label_34 = ClickQLabel(self.verticalLayoutWidget_3)
        self.label_34.setText("")
        self.label_34.setPixmap(QtGui.QPixmap(":/Assets/bg.png"))
        self.label_34.setScaledContents(True)
        self.label_34.setObjectName("label_34")
        self.gridLayout.addWidget(self.label_34, 3, 4, 1, 1)
        self.label_20 = ClickQLabel(self.verticalLayoutWidget_3)
        self.label_20.setText("")
        self.label_20.setPixmap(QtGui.QPixmap(":/Assets/bg.png"))
        self.label_20.setScaledContents(True)
        self.label_20.setObjectName("label_20")
        self.gridLayout.addWidget(self.label_20, 2, 0, 1, 1)
        self.label_00 = ClickQLabel(self.verticalLayoutWidget_3)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Ignored)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_00.sizePolicy().hasHeightForWidth())
        self.label_00.setSizePolicy(sizePolicy)
        self.label_00.setText("")
        self.label_00.setPixmap(QtGui.QPixmap(":/Assets/bg.png"))
        self.label_00.setScaledContents(True)
        self.label_00.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.label_00.setObjectName("label_00")
        self.gridLayout.addWidget(self.label_00, 0, 0, 1, 1)
        self.label_13 = ClickQLabel(self.verticalLayoutWidget_3)
        self.label_13.setText("")
        self.label_13.setPixmap(QtGui.QPixmap(":/Assets/bg.png"))
        self.label_13.setScaledContents(True)
        self.label_13.setObjectName("label_13")
        self.gridLayout.addWidget(self.label_13, 1, 3, 1, 1)
        self.label_15 = ClickQLabel(self.verticalLayoutWidget_3)
        self.label_15.setText("")
        self.label_15.setPixmap(QtGui.QPixmap(":/Assets/bg.png"))
        self.label_15.setScaledContents(True)
        self.label_15.setObjectName("label_15")
        self.gridLayout.addWidget(self.label_15, 1, 5, 1, 1)
        self.label_12 = ClickQLabel(self.verticalLayoutWidget_3)
        self.label_12.setText("")
        self.label_12.setPixmap(QtGui.QPixmap(":/Assets/bg.png"))
        self.label_12.setScaledContents(True)
        self.label_12.setObjectName("label_12")
        self.gridLayout.addWidget(self.label_12, 1, 2, 1, 1)
        self.label_07 = ClickQLabel(self.verticalLayoutWidget_3)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Ignored)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_07.sizePolicy().hasHeightForWidth())
        self.label_07.setSizePolicy(sizePolicy)
        self.label_07.setText("")
        self.label_07.setPixmap(QtGui.QPixmap(":/Assets/bg.png"))
        self.label_07.setScaledContents(True)
        self.label_07.setObjectName("label_07")
        self.gridLayout.addWidget(self.label_07, 0, 7, 1, 1)
        self.label_16 = ClickQLabel(self.verticalLayoutWidget_3)
        self.label_16.setText("")
        self.label_16.setPixmap(QtGui.QPixmap(":/Assets/bg.png"))
        self.label_16.setScaledContents(True)
        self.label_16.setObjectName("label_16")
        self.gridLayout.addWidget(self.label_16, 1, 6, 1, 1)
        self.label_25 = ClickQLabel(self.verticalLayoutWidget_3)
        self.label_25.setText("")
        self.label_25.setPixmap(QtGui.QPixmap(":/Assets/bg.png"))
        self.label_25.setScaledContents(True)
        self.label_25.setObjectName("label_25")
        self.gridLayout.addWidget(self.label_25, 2, 5, 1, 1)
        self.label_01 = ClickQLabel(self.verticalLayoutWidget_3)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Ignored)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_01.sizePolicy().hasHeightForWidth())
        self.label_01.setSizePolicy(sizePolicy)
        self.label_01.setText("")
        self.label_01.setPixmap(QtGui.QPixmap(":/Assets/bg.png"))
        self.label_01.setScaledContents(True)
        self.label_01.setObjectName("label_01")
        self.gridLayout.addWidget(self.label_01, 0, 1, 1, 1)
        self.label_03 = ClickQLabel(self.verticalLayoutWidget_3)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Ignored)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_03.sizePolicy().hasHeightForWidth())
        self.label_03.setSizePolicy(sizePolicy)
        self.label_03.setText("")
        self.label_03.setPixmap(QtGui.QPixmap(":/Assets/bg.png"))
        self.label_03.setScaledContents(True)
        self.label_03.setObjectName("label_03")
        self.gridLayout.addWidget(self.label_03, 0, 3, 1, 1)
        self.label_04 = ClickQLabel(self.verticalLayoutWidget_3)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Ignored)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_04.sizePolicy().hasHeightForWidth())
        self.label_04.setSizePolicy(sizePolicy)
        self.label_04.setText("")
        self.label_04.setPixmap(QtGui.QPixmap(":/Assets/bg.png"))
        self.label_04.setScaledContents(True)
        self.label_04.setObjectName("label_04")
        self.gridLayout.addWidget(self.label_04, 0, 4, 1, 1)
        self.label_11 = ClickQLabel(self.verticalLayoutWidget_3)
        self.label_11.setText("")
        self.label_11.setPixmap(QtGui.QPixmap(":/Assets/bg.png"))
        self.label_11.setScaledContents(True)
        self.label_11.setObjectName("label_11")
        self.gridLayout.addWidget(self.label_11, 1, 1, 1, 1)
        self.label_21 = ClickQLabel(self.verticalLayoutWidget_3)
        self.label_21.setText("")
        self.label_21.setPixmap(QtGui.QPixmap(":/Assets/bg.png"))
        self.label_21.setScaledContents(True)
        self.label_21.setObjectName("label_21")
        self.gridLayout.addWidget(self.label_21, 2, 1, 1, 1)
        self.label_06 = ClickQLabel(self.verticalLayoutWidget_3)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Ignored)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_06.sizePolicy().hasHeightForWidth())
        self.label_06.setSizePolicy(sizePolicy)
        self.label_06.setText("")
        self.label_06.setPixmap(QtGui.QPixmap(":/Assets/bg.png"))
        self.label_06.setScaledContents(True)
        self.label_06.setObjectName("label_06")
        self.gridLayout.addWidget(self.label_06, 0, 6, 1, 1)
        self.label_26 = ClickQLabel(self.verticalLayoutWidget_3)
        self.label_26.setText("")
        self.label_26.setPixmap(QtGui.QPixmap(":/Assets/bg.png"))
        self.label_26.setScaledContents(True)
        self.label_26.setObjectName("label_26")
        self.gridLayout.addWidget(self.label_26, 2, 6, 1, 1)
        self.label_30 = ClickQLabel(self.verticalLayoutWidget_3)
        self.label_30.setText("")
        self.label_30.setPixmap(QtGui.QPixmap(":/Assets/bg.png"))
        self.label_30.setScaledContents(True)
        self.label_30.setObjectName("label_30")
        self.gridLayout.addWidget(self.label_30, 3, 0, 1, 1)
        self.label_05 = ClickQLabel(self.verticalLayoutWidget_3)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Ignored)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_05.sizePolicy().hasHeightForWidth())
        self.label_05.setSizePolicy(sizePolicy)
        self.label_05.setText("")
        self.label_05.setPixmap(QtGui.QPixmap(":/Assets/bg.png"))
        self.label_05.setScaledContents(True)
        self.label_05.setObjectName("label_05")
        self.gridLayout.addWidget(self.label_05, 0, 5, 1, 1)
        self.label_31 = ClickQLabel(self.verticalLayoutWidget_3)
        self.label_31.setText("")
        self.label_31.setPixmap(QtGui.QPixmap(":/Assets/bg.png"))
        self.label_31.setScaledContents(True)
        self.label_31.setObjectName("label_31")
        self.gridLayout.addWidget(self.label_31, 3, 1, 1, 1)
        self.label_32 = ClickQLabel(self.verticalLayoutWidget_3)
        self.label_32.setText("")
        self.label_32.setPixmap(QtGui.QPixmap(":/Assets/bg.png"))
        self.label_32.setScaledContents(True)
        self.label_32.setObjectName("label_32")
        self.gridLayout.addWidget(self.label_32, 3, 2, 1, 1)
        self.label_22 = ClickQLabel(self.verticalLayoutWidget_3)
        self.label_22.setText("")
        self.label_22.setPixmap(QtGui.QPixmap(":/Assets/bg.png"))
        self.label_22.setScaledContents(True)
        self.label_22.setObjectName("label_22")
        self.gridLayout.addWidget(self.label_22, 2, 2, 1, 1)
        self.label_17 = ClickQLabel(self.verticalLayoutWidget_3)
        self.label_17.setText("")
        self.label_17.setPixmap(QtGui.QPixmap(":/Assets/bg.png"))
        self.label_17.setScaledContents(True)
        self.label_17.setObjectName("label_17")
        self.gridLayout.addWidget(self.label_17, 1, 7, 1, 1)
        self.label_02 = ClickQLabel(self.verticalLayoutWidget_3)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Ignored)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_02.sizePolicy().hasHeightForWidth())
        self.label_02.setSizePolicy(sizePolicy)
        self.label_02.setText("")
        self.label_02.setPixmap(QtGui.QPixmap(":/Assets/bg.png"))
        self.label_02.setScaledContents(True)
        self.label_02.setObjectName("label_02")
        self.gridLayout.addWidget(self.label_02, 0, 2, 1, 1)
        self.label_23 = ClickQLabel(self.verticalLayoutWidget_3)
        self.label_23.setText("")
        self.label_23.setPixmap(QtGui.QPixmap(":/Assets/bg.png"))
        self.label_23.setScaledContents(True)
        self.label_23.setObjectName("label_23")
        self.gridLayout.addWidget(self.label_23, 2, 3, 1, 1)
        self.label_14 = ClickQLabel(self.verticalLayoutWidget_3)
        self.label_14.setText("")
        self.label_14.setPixmap(QtGui.QPixmap(":/Assets/bg.png"))
        self.label_14.setScaledContents(True)
        self.label_14.setObjectName("label_14")
        self.gridLayout.addWidget(self.label_14, 1, 4, 1, 1)
        self.label_55 = ClickQLabel(self.verticalLayoutWidget_3)
        self.label_55.setText("")
        self.label_55.setPixmap(QtGui.QPixmap(":/Assets/bg.png"))
        self.label_55.setScaledContents(True)
        self.label_55.setObjectName("label_55")
        self.gridLayout.addWidget(self.label_55, 5, 5, 1, 1)
        self.label_60 = ClickQLabel(self.verticalLayoutWidget_3)
        self.label_60.setText("")
        self.label_60.setPixmap(QtGui.QPixmap(":/Assets/bg.png"))
        self.label_60.setScaledContents(True)
        self.label_60.setObjectName("label_60")
        self.gridLayout.addWidget(self.label_60, 6, 0, 1, 1)
        self.label_61 = ClickQLabel(self.verticalLayoutWidget_3)
        self.label_61.setText("")
        self.label_61.setPixmap(QtGui.QPixmap(":/Assets/bg.png"))
        self.label_61.setScaledContents(True)
        self.label_61.setObjectName("label_61")
        self.gridLayout.addWidget(self.label_61, 6, 1, 1, 1)
        self.label_66 = ClickQLabel(self.verticalLayoutWidget_3)
        self.label_66.setText("")
        self.label_66.setPixmap(QtGui.QPixmap(":/Assets/bg.png"))
        self.label_66.setScaledContents(True)
        self.label_66.setObjectName("label_66")
        self.gridLayout.addWidget(self.label_66, 6, 6, 1, 1)
        self.label_70 = ClickQLabel(self.verticalLayoutWidget_3)
        self.label_70.setText("")
        self.label_70.setPixmap(QtGui.QPixmap(":/Assets/bg.png"))
        self.label_70.setScaledContents(True)
        self.label_70.setObjectName("label_70")
        self.gridLayout.addWidget(self.label_70, 7, 0, 1, 1)
        self.label_72 = ClickQLabel(self.verticalLayoutWidget_3)
        self.label_72.setText("")
        self.label_72.setPixmap(QtGui.QPixmap(":/Assets/bg.png"))
        self.label_72.setScaledContents(True)
        self.label_72.setObjectName("label_72")
        self.gridLayout.addWidget(self.label_72, 7, 2, 1, 1)
        self.label_63 = ClickQLabel(self.verticalLayoutWidget_3)
        self.label_63.setText("")
        self.label_63.setPixmap(QtGui.QPixmap(":/Assets/bg.png"))
        self.label_63.setScaledContents(True)
        self.label_63.setObjectName("label_63")
        self.gridLayout.addWidget(self.label_63, 6, 3, 1, 1)
        self.label_62 = ClickQLabel(self.verticalLayoutWidget_3)
        self.label_62.setText("")
        self.label_62.setPixmap(QtGui.QPixmap(":/Assets/bg.png"))
        self.label_62.setScaledContents(True)
        self.label_62.setObjectName("label_62")
        self.gridLayout.addWidget(self.label_62, 6, 2, 1, 1)
        self.label_51 = ClickQLabel(self.verticalLayoutWidget_3)
        self.label_51.setText("")
        self.label_51.setPixmap(QtGui.QPixmap(":/Assets/bg.png"))
        self.label_51.setScaledContents(True)
        self.label_51.setObjectName("label_51")
        self.gridLayout.addWidget(self.label_51, 5, 1, 1, 1)
        self.label_67 = ClickQLabel(self.verticalLayoutWidget_3)
        self.label_67.setText("")
        self.label_67.setPixmap(QtGui.QPixmap(":/Assets/bg.png"))
        self.label_67.setScaledContents(True)
        self.label_67.setObjectName("label_67")
        self.gridLayout.addWidget(self.label_67, 6, 7, 1, 1)
        self.label_57 = ClickQLabel(self.verticalLayoutWidget_3)
        self.label_57.setText("")
        self.label_57.setPixmap(QtGui.QPixmap(":/Assets/bg.png"))
        self.label_57.setScaledContents(True)
        self.label_57.setObjectName("label_57")
        self.gridLayout.addWidget(self.label_57, 5, 7, 1, 1)
        self.label_52 = ClickQLabel(self.verticalLayoutWidget_3)
        self.label_52.setText("")
        self.label_52.setPixmap(QtGui.QPixmap(":/Assets/bg.png"))
        self.label_52.setScaledContents(True)
        self.label_52.setObjectName("label_52")
        self.gridLayout.addWidget(self.label_52, 5, 2, 1, 1)
        self.label_56 = ClickQLabel(self.verticalLayoutWidget_3)
        self.label_56.setText("")
        self.label_56.setPixmap(QtGui.QPixmap(":/Assets/bg.png"))
        self.label_56.setScaledContents(True)
        self.label_56.setObjectName("label_56")
        self.gridLayout.addWidget(self.label_56, 5, 6, 1, 1)
        self.label_71 = ClickQLabel(self.verticalLayoutWidget_3)
        self.label_71.setText("")
        self.label_71.setPixmap(QtGui.QPixmap(":/Assets/bg.png"))
        self.label_71.setScaledContents(True)
        self.label_71.setObjectName("label_71")
        self.gridLayout.addWidget(self.label_71, 7, 1, 1, 1)
        self.label_54 = ClickQLabel(self.verticalLayoutWidget_3)
        self.label_54.setText("")
        self.label_54.setPixmap(QtGui.QPixmap(":/Assets/bg.png"))
        self.label_54.setScaledContents(True)
        self.label_54.setObjectName("label_54")
        self.gridLayout.addWidget(self.label_54, 5, 4, 1, 1)
        self.label_64 = ClickQLabel(self.verticalLayoutWidget_3)
        self.label_64.setText("")
        self.label_64.setPixmap(QtGui.QPixmap(":/Assets/bg.png"))
        self.label_64.setScaledContents(True)
        self.label_64.setObjectName("label_64")
        self.gridLayout.addWidget(self.label_64, 6, 4, 1, 1)
        self.label_53 = ClickQLabel(self.verticalLayoutWidget_3)
        self.label_53.setText("")
        self.label_53.setPixmap(QtGui.QPixmap(":/Assets/bg.png"))
        self.label_53.setScaledContents(True)
        self.label_53.setObjectName("label_53")
        self.gridLayout.addWidget(self.label_53, 5, 3, 1, 1)
        self.label_65 = ClickQLabel(self.verticalLayoutWidget_3)
        self.label_65.setText("")
        self.label_65.setPixmap(QtGui.QPixmap(":/Assets/bg.png"))
        self.label_65.setScaledContents(True)
        self.label_65.setObjectName("label_65")
        self.gridLayout.addWidget(self.label_65, 6, 5, 1, 1)
        self.label_50 = ClickQLabel(self.verticalLayoutWidget_3)
        self.label_50.setText("")
        self.label_50.setPixmap(QtGui.QPixmap(":/Assets/bg.png"))
        self.label_50.setScaledContents(True)
        self.label_50.setObjectName("label_50")
        self.gridLayout.addWidget(self.label_50, 5, 0, 1, 1)
        self.label_44 = ClickQLabel(self.verticalLayoutWidget_3)
        self.label_44.setText("")
        self.label_44.setPixmap(QtGui.QPixmap(":/Assets/bg.png"))
        self.label_44.setScaledContents(True)
        self.label_44.setObjectName("label_44")
        self.gridLayout.addWidget(self.label_44, 4, 4, 1, 1)
        self.label_36 = ClickQLabel(self.verticalLayoutWidget_3)
        self.label_36.setText("")
        self.label_36.setPixmap(QtGui.QPixmap(":/Assets/bg.png"))
        self.label_36.setScaledContents(True)
        self.label_36.setObjectName("label_36")
        self.gridLayout.addWidget(self.label_36, 3, 6, 1, 1)
        self.label_45 = ClickQLabel(self.verticalLayoutWidget_3)
        self.label_45.setMinimumSize(QtCore.QSize(0, 0))
        self.label_45.setText("")
        self.label_45.setPixmap(QtGui.QPixmap(":/Assets/bg.png"))
        self.label_45.setScaledContents(True)
        self.label_45.setObjectName("label_45")
        self.gridLayout.addWidget(self.label_45, 4, 5, 1, 1)
        self.label_46 = ClickQLabel(self.verticalLayoutWidget_3)
        self.label_46.setText("")
        self.label_46.setPixmap(QtGui.QPixmap(":/Assets/bg.png"))
        self.label_46.setScaledContents(True)
        self.label_46.setObjectName("label_46")
        self.gridLayout.addWidget(self.label_46, 4, 6, 1, 1)
        self.label_47 = ClickQLabel(self.verticalLayoutWidget_3)
        self.label_47.setText("")
        self.label_47.setPixmap(QtGui.QPixmap(":/Assets/bg.png"))
        self.label_47.setScaledContents(True)
        self.label_47.setObjectName("label_47")
        self.gridLayout.addWidget(self.label_47, 4, 7, 1, 1)
        self.label_35 = ClickQLabel(self.verticalLayoutWidget_3)
        self.label_35.setText("")
        self.label_35.setPixmap(QtGui.QPixmap(":/Assets/bg.png"))
        self.label_35.setScaledContents(True)
        self.label_35.setObjectName("label_35")
        self.gridLayout.addWidget(self.label_35, 3, 5, 1, 1)
        self.label_41 = ClickQLabel(self.verticalLayoutWidget_3)
        self.label_41.setText("")
        self.label_41.setPixmap(QtGui.QPixmap(":/Assets/bg.png"))
        self.label_41.setScaledContents(True)
        self.label_41.setObjectName("label_41")
        self.gridLayout.addWidget(self.label_41, 4, 1, 1, 1)
        self.label_37 = ClickQLabel(self.verticalLayoutWidget_3)
        self.label_37.setText("")
        self.label_37.setPixmap(QtGui.QPixmap(":/Assets/bg.png"))
        self.label_37.setScaledContents(True)
        self.label_37.setObjectName("label_37")
        self.gridLayout.addWidget(self.label_37, 3, 7, 1, 1)
        self.label_40 = ClickQLabel(self.verticalLayoutWidget_3)
        self.label_40.setText("")
        self.label_40.setPixmap(QtGui.QPixmap(":/Assets/bg.png"))
        self.label_40.setScaledContents(True)
        self.label_40.setObjectName("label_40")
        self.gridLayout.addWidget(self.label_40, 4, 0, 1, 1)
        self.label_42 = ClickQLabel(self.verticalLayoutWidget_3)
        self.label_42.setText("")
        self.label_42.setPixmap(QtGui.QPixmap(":/Assets/bg.png"))
        self.label_42.setScaledContents(True)
        self.label_42.setObjectName("label_42")
        self.gridLayout.addWidget(self.label_42, 4, 2, 1, 1)
        self.label_43 = ClickQLabel(self.verticalLayoutWidget_3)
        self.label_43.setText("")
        self.label_43.setPixmap(QtGui.QPixmap(":/Assets/bg.png"))
        self.label_43.setScaledContents(True)
        self.label_43.setObjectName("label_43")
        self.gridLayout.addWidget(self.label_43, 4, 3, 1, 1)
        self.label_75 = ClickQLabel(self.verticalLayoutWidget_3)
        self.label_75.setText("")
        self.label_75.setPixmap(QtGui.QPixmap(":/Assets/bg.png"))
        self.label_75.setScaledContents(True)
        self.label_75.setObjectName("label_75")
        self.gridLayout.addWidget(self.label_75, 7, 5, 1, 1)
        self.label_76 = ClickQLabel(self.verticalLayoutWidget_3)
        self.label_76.setText("")
        self.label_76.setPixmap(QtGui.QPixmap(":/Assets/bg.png"))
        self.label_76.setScaledContents(True)
        self.label_76.setObjectName("label_76")
        self.gridLayout.addWidget(self.label_76, 7, 6, 1, 1)
        self.label_77 = ClickQLabel(self.verticalLayoutWidget_3)
        self.label_77.setText("")
        self.label_77.setPixmap(QtGui.QPixmap(":/Assets/bg.png"))
        self.label_77.setScaledContents(True)
        self.label_77.setObjectName("label_77")
        self.gridLayout.addWidget(self.label_77, 7, 7, 1, 1)
        self.label_74 = ClickQLabel(self.verticalLayoutWidget_3)
        self.label_74.setText("")
        self.label_74.setPixmap(QtGui.QPixmap(":/Assets/bg.png"))
        self.label_74.setScaledContents(True)
        self.label_74.setObjectName("label_74")
        self.gridLayout.addWidget(self.label_74, 7, 4, 1, 1)
        self.label_73 = ClickQLabel(self.verticalLayoutWidget_3)
        self.label_73.setText("")
        self.label_73.setPixmap(QtGui.QPixmap(":/Assets/bg.png"))
        self.label_73.setScaledContents(True)
        self.label_73.setObjectName("label_73")
        self.gridLayout.addWidget(self.label_73, 7, 3, 1, 1)
        self.gridLayout.setColumnMinimumWidth(0, 64)
        self.gridLayout.setColumnMinimumWidth(1, 64)
        self.gridLayout.setColumnMinimumWidth(2, 64)
        self.gridLayout.setColumnMinimumWidth(3, 64)
        self.gridLayout.setColumnMinimumWidth(4, 64)
        self.gridLayout.setColumnMinimumWidth(5, 64)
        self.gridLayout.setColumnMinimumWidth(6, 64)
        self.gridLayout.setColumnMinimumWidth(7, 64)
        self.gridLayout.setRowMinimumHeight(0, 64)
        self.gridLayout.setRowMinimumHeight(1, 64)
        self.gridLayout.setRowMinimumHeight(2, 64)
        self.gridLayout.setRowMinimumHeight(3, 64)
        self.gridLayout.setRowMinimumHeight(4, 64)
        self.gridLayout.setRowMinimumHeight(5, 64)
        self.gridLayout.setRowMinimumHeight(6, 64)
        self.gridLayout.setRowMinimumHeight(7, 64)
        self.gridLayout.setColumnStretch(0, 1)
        self.gridLayout.setColumnStretch(1, 1)
        self.gridLayout.setColumnStretch(2, 1)
        self.gridLayout.setColumnStretch(3, 1)
        self.gridLayout.setColumnStretch(4, 1)
        self.gridLayout.setColumnStretch(5, 1)
        self.gridLayout.setColumnStretch(6, 1)
        self.gridLayout.setColumnStretch(7, 1)
        self.gridLayout.setRowStretch(0, 1)
        self.gridLayout.setRowStretch(1, 1)
        self.gridLayout.setRowStretch(2, 1)
        self.gridLayout.setRowStretch(3, 1)
        self.gridLayout.setRowStretch(4, 1)
        self.gridLayout.setRowStretch(5, 1)
        self.gridLayout.setRowStretch(6, 1)
        self.gridLayout.setRowStretch(7, 1)
        self.verticalLayout_3.addLayout(self.gridLayout)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setContentsMargins(-1, 10, -1, 10)
        self.horizontalLayout.setObjectName("horizontalLayout")
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem)
        self.firstPageButton = QtWidgets.QPushButton(self.verticalLayoutWidget_3)
        self.firstPageButton.setMaximumSize(QtCore.QSize(45, 16777215))
        self.firstPageButton.setObjectName("firstPageButton")
        self.horizontalLayout.addWidget(self.firstPageButton)
        self.lastPageButton = QtWidgets.QPushButton(self.verticalLayoutWidget_3)
        self.lastPageButton.setEnabled(True)
        self.lastPageButton.setMinimumSize(QtCore.QSize(0, 0))
        self.lastPageButton.setMaximumSize(QtCore.QSize(45, 16777215))
        self.lastPageButton.setObjectName("lastPageButton")
        self.horizontalLayout.addWidget(self.lastPageButton)
        self.page = QtWidgets.QLabel(self.verticalLayoutWidget_3)
        self.page.setObjectName("page")
        self.horizontalLayout.addWidget(self.page)
        self.slash = QtWidgets.QLabel(self.verticalLayoutWidget_3)
        self.slash.setObjectName("slash")
        self.horizontalLayout.addWidget(self.slash)
        self.pageTotal = QtWidgets.QLabel(self.verticalLayoutWidget_3)
        self.pageTotal.setObjectName("pageTotal")
        self.horizontalLayout.addWidget(self.pageTotal)
        self.nextPageButton = QtWidgets.QPushButton(self.verticalLayoutWidget_3)
        self.nextPageButton.setMaximumSize(QtCore.QSize(45, 16777215))
        self.nextPageButton.setObjectName("nextPageButton")
        self.horizontalLayout.addWidget(self.nextPageButton)
        self.finalPageButton = QtWidgets.QPushButton(self.verticalLayoutWidget_3)
        self.finalPageButton.setMaximumSize(QtCore.QSize(45, 16777215))
        self.finalPageButton.setObjectName("finalPageButton")
        self.horizontalLayout.addWidget(self.finalPageButton)
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem1)
        self.verticalLayout_3.addLayout(self.horizontalLayout)
        self.verticalLayoutWidget = QtWidgets.QWidget(self.splitter)
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.label = MoveQLabel(self.verticalLayoutWidget)
        self.label.setMinimumSize(QtCore.QSize(768, 768))
        self.label.setText("")
        self.label.setPixmap(QtGui.QPixmap(":/Assets/bg.png"))
        self.label.setScaledContents(True)
        self.label.setObjectName("label")
        self.verticalLayout.addWidget(self.label)
        self.position = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.position.setIndent(5)
        self.position.setObjectName("position")
        self.verticalLayout.addWidget(self.position)
        spacerItem2 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout.addItem(spacerItem2)
        self.gridLayout_2.addWidget(self.splitter, 1, 0, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1836, 34))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.modelPath.setText(_translate("MainWindow", "Model Path ..."))
        self.loadModelButton.setText(_translate("MainWindow", "LoadModel"))
        self.firstPageButton.setText(_translate("MainWindow", "<<"))
        self.lastPageButton.setText(_translate("MainWindow", "<"))
        self.page.setText(_translate("MainWindow", "0"))
        self.slash.setText(_translate("MainWindow", "/"))
        self.pageTotal.setText(_translate("MainWindow", "0"))
        self.nextPageButton.setText(_translate("MainWindow", ">"))
        self.finalPageButton.setText(_translate("MainWindow", ">>"))
        self.position.setText(_translate("MainWindow", "X: 0 Y: 0 V: 0"))

import resource_rc
