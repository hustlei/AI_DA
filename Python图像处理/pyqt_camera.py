# -*- coding: utf-8 -*-
"""
Created on Fri Feb 18 20:18:05 2022

@author: lei
"""

import time, sys
from PyQt5 import QtWidgets, QtMultimediaWidgets
from PyQt5.QtMultimedia import QCamera, QCameraImageCapture, QCameraViewfinderSettings


class CameraMainWin(QtWidgets.QMainWindow):
    def __init__(self):
        super(CameraMainWin, self).__init__()
        self.setupUi(self)
        # 定义相机实例对象并设置捕获模式
        self.camera = QCamera()
        self.camera.setCaptureMode(QCamera.CaptureViewfinder)
        self.cameraOpened = False  # 设置相机打开状态为未打开
        # 设置取景器分辨率
        viewFinderSettings = QCameraViewfinderSettings()
        viewFinderSettings.setResolution(800, 600)
        self.camera.setViewfinderSettings(viewFinderSettings)
        # 初始化取景器
        self.viewCamera = QtMultimediaWidgets.QCameraViewfinder(self)
        self.camera.setViewfinder(self.viewCamera)
        self.camerLayout.addWidget(self.viewCamera)  # 取景器放置到预留的布局中
        # 设置图像捕获
        self.capImg = QCameraImageCapture(self.camera)
        self.capImg.setCaptureDestination(
            QCameraImageCapture.CaptureToFile
        )  # CaptureToBuffer

    def 

    # 相机（摄像头）开关处理
    def switchCamera(self):
        if not self.cameraOpened:
            self.camera.start()
            self.cameraOpened = True
            self.btnSwitchCamera.setText("关闭摄像头")
        else:
            self.camera.stop()
            self.cameraOpened = False
            self.btnSwitchCamera.setText("打开摄像头")

    def takePic(self):  # 拍照响应槽函数，照片保存到文件
        FName = rf"c:\temp\capimg\cap{time.strftime('%Y%m%d%H%M%S', time.localtime())}"  # 文件名初始化
        self.capImg.capture(FName)
        print(f"捕获图像保存到文件：{FName}.jpg")


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    W = CameraMainWin()
    W.show()
    sys.exit(app.exec_())
