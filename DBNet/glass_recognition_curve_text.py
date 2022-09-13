from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QTime, QTimer
from PyQt5.QtWidgets import *
from PIL import Image, ImageQt
import sys, os, cv2
# from PyQt5.Qt import *
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtWidgets import QWidget
from PyQt5.QtGui import QImage, QPixmap, QWindow
import pyttsx3
import os
from playsound import playsound


class Ui_mainWindow(object):
    def setupUi(self, mainWindow):
        mainWindow.setObjectName("MainWindow")
        mainWindow.resize(667, 803)

        self.centralwidget = QtWidgets.QWidget(mainWindow)
        self.centralwidget.setObjectName("centralwidget")

        # self.imagePath = QtWidgets.QLabel(self.centralwidget)
        # self.imagePath.setGeometry(QtCore.QRect(170, 18, 461, 41))
        # self.imagePath.setFrameShape(QtWidgets.QFrame.Panel)
        # self.imagePath.setText("")
        # self.imagePath.setObjectName("imagePath")

        self.result_label = QtWidgets.QLabel(self.centralwidget)
        self.result_label.setGeometry(QtCore.QRect(30, 500, 141, 31))  # result position
        font = QtGui.QFont()
        font.setPointSize(18)
        self.result_label.setFont(font)
        self.result_label.setFrameShape(QtWidgets.QFrame.Box)
        self.result_label.setAlignment(QtCore.Qt.AlignCenter)
        self.result_label.setObjectName("result_label")

        self.cameraArea = QtWidgets.QLabel(self.centralwidget)
        self.cameraArea.setGeometry(QtCore.QRect(40, 90, 571, 331))
        self.cameraArea.setFrameShape(QtWidgets.QFrame.Box)
        self.cameraArea.setText("")
        self.cameraArea.setAlignment(QtCore.Qt.AlignCenter)
        self.cameraArea.setObjectName("cameraArea")

        self.resultArea = QtWidgets.QLabel(self.centralwidget)
        self.resultArea.setGeometry(QtCore.QRect(30, 530, 591, 161))
        self.resultArea.setFrameShape(QtWidgets.QFrame.Box)
        self.resultArea.setText("")
        self.resultArea.setWordWrap(True)
        self.resultArea.setObjectName("resultArea")

        self.imageBtn = QtWidgets.QPushButton(self.centralwidget)
        self.imageBtn.setGeometry(QtCore.QRect(39, 18, 570, 60))
        font = QtGui.QFont()
        font.setPointSize(18)
        self.imageBtn.setFont(font)
        self.imageBtn.setObjectName("imageBtn")
        self.imageBtn.clicked.connect(self.selectImage)

        self.open_camera = QtWidgets.QPushButton(self.centralwidget)
        self.open_camera.setGeometry(QtCore.QRect(35, 430, 131, 51))
        self.open_camera.setObjectName("open_camera")
        self.open_camera.clicked.connect(self.openCamera)

        self.capture_button = QtWidgets.QPushButton(self.centralwidget)
        self.capture_button.setGeometry(QtCore.QRect(260, 430, 131, 51))
        self.capture_button.setObjectName("capture_button")
        self.capture_button.clicked.connect(self.capture)

        self.close_camera = QtWidgets.QPushButton(self.centralwidget)
        self.close_camera.setGeometry(QtCore.QRect(480, 430, 131, 51))
        self.close_camera.setObjectName("close_camera")
        self.close_camera.clicked.connect(self.closeCamera)

        mainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(mainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 667, 22))
        self.menubar.setObjectName("menubar")
        mainWindow.setMenuBar(self.menubar)

        self.statusbar = QtWidgets.QStatusBar(mainWindow)
        self.statusbar.setObjectName("statusbar")
        mainWindow.setStatusBar(self.statusbar)
        self.toolBar = QtWidgets.QToolBar(mainWindow)
        self.toolBar.setObjectName("toolBar")
        mainWindow.addToolBar(QtCore.Qt.RightToolBarArea, self.toolBar)
        self.retranslateUi(mainWindow)
        QtCore.QMetaObject.connectSlotsByName(mainWindow)




    def retranslateUi(self, mainWindow):
        _translate = QtCore.QCoreApplication.translate
        mainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))

        self.result_label.setText(_translate("MainWindow", "Result"))
        self.imageBtn.setText(_translate("MainWindow", "Jasmine's Curve Text Recognition"))
        # self.start_reading_button.setText(_translate("MainWindow", "Start Reading"))
        self.open_camera.setText(_translate("MainWindow", "Open Camera"))
        self.capture_button.setText(_translate("MainWindow", "Recognition"))
        self.close_camera.setText(_translate("MainWindow", "Close Camera"))
        self.toolBar.setWindowTitle(_translate("MainWindow", "toolBar"))




class Window(Ui_mainWindow,QMainWindow):
    # i = 3
    def __init__(self):
        super().__init__()
        print("初始化")
        self.initUI()
        self.initArgs()
        self.timer.timeout.connect(self.showFrame)

        self.setupUi(self)

    def setup_ui(self):
        pass

    def binding(self):
        pass

    def initUI(self):
        self.win = QMainWindow()
        self.setupUi(self.win)

    def initArgs(self):
        self.timer = QTimer()
        self.cap = None
        self.flag = False
        self.frame = None

    def initSlot(self):
        print(' ')

    def openCamera(self):
        print('open camera')
        self.cap = cv2.VideoCapture(0)  # 0是表示调用电脑自带的摄像头，1是表示调用外接摄像头
        self.timer.start(100)

    def selectImage(self):
        '''
        function: 上传一张电脑上的图片
        '''
        imageName, imageType = QFileDialog.getOpenFileName(
            self,
            'select image',
            os.getcwd(),
        )
        if imageName == '':
            msg = (QMessageBox.warning(self, 'Warning', 'Please select an image', QMessageBox.Ok, QMessageBox.Ok))
        else:
            self.imagePath.setText(imageName)
            self.showImage(imageName)
            # self.inference(self.frame)

    def showImage(self, path):
        '''
        function: 根据图片的大小进行相应的调整来将一个图片完整的现实在display框里
        parameters: path = 路径
        '''
        img = Image.open(path)
        w, h = img.size
        if w > 421 and h < 301:  # 421是display的宽度，301是display的高度
            r = w / 421
            img = img.resize((421, int(h / r)), 4)
        elif h > 301 and w < 481:
            r = h / 301
            img = img.resize((int(w / r), 281), 4)
        elif h > 301 and w > 481:
            r = h / 301
            img = img.resize((int(w / r), 281), 4)
        img = ImageQt.toqpixmap(img)
        self.frame = self.cameraArea.setPixmap(img)

    def showFrame(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.resize(frame, (591, 332))
            self.frame = frame
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = QImage(frame.data, frame.shape[1], frame.shape[0], frame.shape[1] * 3, QImage.Format_RGB888)
            self.cameraArea.setPixmap(QPixmap.fromImage(frame))

    def capture(self):
        self.flag = True
        self.closeCamera()
        self.inference(self.frame)
        self.flag = False

    def closeCamera(self):
        self.timer.stop()
        if not self.flag:
            self.cameraArea.clear()
            self.cap.release()

    def inference(self, frame):
        cv2.imwrite('./recog/test_imgs/tmp.jpg', frame)
        os.system('python inference.py')
        with open('result.txt', 'r', encoding='utf8') as f:
            lines = f.readlines()
            lines = [line.strip('\n') for line in lines]
            s = ''
            for line in lines: s += line + ' '
        self.resultArea.setText(s)

if __name__ == '__main__':
    import sys

    app = QtWidgets.QApplication(sys.argv)
    window = Window()
    window.show()
    sys.exit(app.exec_())

