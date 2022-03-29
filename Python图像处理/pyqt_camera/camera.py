#!/usr/bin/env python
# coding=utf-8
"""
Translating to python by Filipp Kucheryavy aka Frizzy <filipp.s.frizzy@gmail.com>

/****************************************************************************
**
** Copyright (C) 2013 Digia Plc and/or its subsidiary(-ies).
** Contact: http://www.qt-project.org/legal
**
** This file is part of the examples of the Qt Toolkit.
**
** $QT_BEGIN_LICENSE:BSD$
** You may use this file under the terms of the BSD license as follows:
**
** "Redisix.text_typeibution and use in source and binary forms, with or without
** modification, are permitted provided that the following conditions are
** met:
**   * Redistributions of source code must retain the above copyright
**     notice, this list of conditions and the following disclaimer.
**   * Redistributions in binary form must reproduce the above copyright
**     notice, this list of conditions and the following disclaimer in
**     the documentation and/or other materials provided with the
**     distribution.
**   * Neither the name of Digia Plc and its Subsidiary(-ies) nor the names
**     of its contributors may be used to endorse or promote products derived
**     from this software without specific prior written permission.
**
**
** THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
** "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
** LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
** A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
** OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
** SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
** LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
** DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
** THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
** (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
** OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE."
**
** $QT_END_LICENSE$
**
****************************************************************************/
"""

import sys
import six

from videosettings import VideoSettings
from imagesettings import ImageSettings
from camera_ui import Ui_Camera

try:
    from PySide import QtWidgets, QtCore, QtGui, QtMultimedia, QtMultimediaWidgets, uic
except:
    try:
        from PyQt5 import (
            QtWidgets,
            QtCore,
            QtGui,
            QtMultimedia,
            QtMultimediaWidgets,
            uic,
        )
    except:
        six.print_(
            ("Error: can't load PySide or PyQT"), file=sys.stderr, end="\n", sep=" "
        )
        sys.exit()


class Camera(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        QtWidgets.QMainWindow.__init__(self, parent)

        self.camera = QtMultimedia.QCamera()

        self.imageCapture = QtMultimedia.QCameraImageCapture(self.camera, self)
        self.mediaRecorder = QtMultimedia.QMediaRecorder(self.camera, self)

        self.imageSettings = QtMultimedia.QImageEncoderSettings()
        self.audioSettings = QtMultimedia.QAudioEncoderSettings()
        self.videoSettings = QtMultimedia.QVideoEncoderSettings()
        self.videoContainerFormat = ""
        self.isCapturingImage = False
        self.applicationExiting = False

        # Set up the user interface from Designer.
        # self.ui = uic.loadUi('camera.ui', self)
        self.ui = Ui_Camera()
        self.ui.setupUi(self)

        # Camera devices:
        self.cameraDevice = QtCore.QByteArray()

        self.videoDevicesGroup = QtWidgets.QActionGroup(self)
        self.videoDevicesGroup.setExclusive(True)
        for deviceName in self.camera.availableDevices():
            description = self.camera.deviceDescription(deviceName)
            videoDeviceAction = QtWidgets.QAction(description, self.videoDevicesGroup)
            videoDeviceAction.setCheckable(True)
            videoDeviceAction.setData(QtCore.QVariant(deviceName))
            if self.cameraDevice.isEmpty():
                cameraDevice = deviceName
                videoDeviceAction.setChecked(True)
            self.ui.menuDevices.addAction(videoDeviceAction)

        self.videoDevicesGroup.triggered.connect(self.updateCameraDevice)
        self.ui.captureWidget.currentChanged["int"].connect(self.updateCaptureMode)

        try:
            self.ui.lockButton.hide()
        except:
            pass

        self.setCamera(self.cameraDevice)

    def setCamera(self, cameraDevice):
        if cameraDevice.isEmpty():
            self.camera = QtMultimedia.QCamera()
        else:
            self.camera = QtMultimedia.QCamera(cameraDevice)

        self.camera.stateChanged.connect(self.updateCameraState)
        self.camera.error.connect(self.displayCameraError)

        self.mediaRecorder = QtMultimedia.QMediaRecorder(self.camera)
        self.mediaRecorder.stateChanged.connect(self.updateRecorderState)

        self.imageCapture = QtMultimedia.QCameraImageCapture(self.camera)

        self.mediaRecorder.durationChanged.connect(self.updateRecordTime)
        self.mediaRecorder.error.connect(self.displayRecorderError)

        self.mediaRecorder.setMetaData(
            QtMultimedia.QMediaMetaData.Title,
            QtCore.QVariant(six.text_type("Test Title")),
        )

        self.ui.exposureCompensation.valueChanged["int"].connect(
            self.setExposureCompensation
        )

        self.camera.setViewfinder(self.ui.viewfinder)

        self.updateCameraState(self.camera.state())
        self.updateLockStatus(
            self.camera.lockStatus(), QtMultimedia.QCamera().UserRequest
        )
        self.updateRecorderState(self.mediaRecorder.state())

        self.imageCapture.readyForCaptureChanged.connect(self.readyForCapture)
        self.imageCapture.imageCaptured.connect(self.processCapturedImage)
        self.imageCapture.imageSaved.connect(self.imageSaved)

        self.camera.lockStatusChanged.connect(self.updateLockStatus)

        self.ui.captureWidget.setTabEnabled(
            0,
            (
                self.camera.isCaptureModeSupported(
                    QtMultimedia.QCamera().CaptureStillImage
                )
            ),
        )
        self.ui.captureWidget.setTabEnabled(
            1, (self.camera.isCaptureModeSupported(QtMultimedia.QCamera().CaptureVideo))
        )

        self.updateCaptureMode()
        self.camera.start()
        print(self.camera.supportedViewfinderResolutions())

    def startCamera(self):
        self.camera.start()

    def stopCamera(self):
        self.camera.stop()

    def record(self):
        self.mediaRecorder.record()
        self.updateRecordTime()

    def pause(self):
        self.mediaRecorder.pause()

    def stop(self):
        self.mediaRecorder.stop()

    def setMuted(self, muted):
        self.mediaRecorder.setMuted(muted)

    def toggleLock(self):
        if self.camera.lockStatus() == QtMultimedia.QCamera().Searching:
            pass
        if self.camera.lockStatus() == QtMultimedia.QCamera().Locked:
            self.camera.unlock()

        if self.camera.lockStatus() == QtMultimedia.QCamera().Unlocked:
            self.camera.searchAndLock()

    def takeImage(self):
        self.isCapturingImage = True
        self.imageCapture.capture()

    def configureCaptureSettings(self):
        if self.camera.captureMode() == QtMultimedia.QCamera().CaptureStillImage:
            self.configureImageSettings()

        elif self.camera.captureMode() == QtMultimedia.QCamera().CaptureVideo:
            self.configureVideoSettings()

    def configureVideoSettings(self):
        settingsDialog = VideoSettings(self.mediaRecorder)

        settingsDialog.setAudioSettings(self.audioSettings)
        settingsDialog.setVideoSettings(self.videoSettings)
        settingsDialog.setFormat(self.videoContainerFormat)

        if settingsDialog.exec():
            self.audioSettings = settingsDialog.audioSettings
            self.videoSettings = settingsDialog.videoSettings
            self.videoContainerFormat = settingsDialog.format()

            self.mediaRecorder.setEncodingSettings(
                self.audioSettings, self.videoSettings, self.videoContainerFormat
            )

    def configureImageSettings(self):
        settingsDialog = ImageSettings(self.imageCapture)

        settingsDialog.setImageSettings(self.imageSettings)

        if settingsDialog.exec():
            self.imageSettings = settingsDialog.imageSettings()
            self.imageCapture.setEncodingSettings(self.imageSettings)

    def displayRecorderError(self):
        QtWidgets.QMessageBox().warning(
            self, "Capture error", self.mediaRecorder.errorString()
        )

    def displayCameraError(self):
        QtWidgets.QMessageBox().warning(self, "Camera error", self.camera.errorString())

    def updateCameraDevice(self, action):
        six.print_((action.data()), file=sys.stderr, end="\n", sep=" ")
        self.setCamera(action.data())

    def updateCameraState(self, state):
        if state == QtMultimedia.QCamera().ActiveState:
            self.ui.actionStartCamera.setEnabled(False)
            self.ui.actionStopCamera.setEnabled(True)
            self.ui.captureWidget.setEnabled(True)
            self.ui.actionSettings.setEnabled(True)

        if state == QtMultimedia.QCamera().LoadedState:
            self.ui.actionStartCamera.setEnabled(True)
            self.ui.actionStopCamera.setEnabled(False)
            self.ui.captureWidget.setEnabled(False)
            self.ui.actionSettings.setEnabled(False)

    def updateCaptureMode(self):
        tabIndex = self.ui.captureWidget.currentIndex()
        if tabIndex == 0:
            captureMode = QtMultimedia.QCamera().CaptureStillImage
        else:
            captureMode = QtMultimedia.QCamera().CaptureVideo

        if self.camera.isCaptureModeSupported(captureMode):
            self.camera.setCaptureMode(captureMode)

    def updateRecorderState(self, state):
        if state == self.mediaRecorder.StoppedState:
            self.ui.recordButton.setEnabled(True)
            self.ui.pauseButton.setEnabled(True)
            self.ui.stopButton.setEnabled(False)

        if state == self.mediaRecorder.PausedState:
            self.ui.recordButton.setEnabled(True)
            self.ui.pauseButton.setEnabled(False)
            self.ui.stopButton.setEnabled(True)

        if state == self.mediaRecorder.RecordingState:
            self.ui.recordButton.setEnabled(False)
            self.ui.pauseButton.setEnabled(True)
            self.ui.stopButton.setEnabled(True)

    def setExposureCompensation(self, index):
        self.camera.exposure().setExposureCompensation(index * 0.5)

    def updateRecordTime(self):
        string = six.text_type("Recorded %f sec") % (
            (self.mediaRecorder.duration() / 1000)
        )
        self.ui.statusbar.showMessage(string)

    def processCapturedImage(self, requestId, img):
        # Q_UNUSED(requestId);
        scaledImage = img.scaled(self.ui.viewfinder.size(), 1, 1)
        self.ui.lastImagePreviewLabel.setPixmap(QtGui.QPixmap().fromImage(scaledImage))

        # Display captured image for 4 seconds.
        self.displayCapturedImage()
        QtCore.QTimer().singleShot(4000, self.displayViewfinder)

    def updateLockStatus(self, status, reason):
        indicationColor = QtGui.QColor(QtCore.Qt.black)

        if status == QtMultimedia.QCamera().Searching:
            indicationColor = QtCore.Qt.yellow
            self.ui.statusbar.showMessage("Focusing...")
            self.ui.lockButton.setText("Focusing...")

        if status == QtMultimedia.QCamera().Locked:
            indicationColor = QtCore.Qt.darkGreen
            self.ui.lockButton.setText("Unlock")
            self.ui.statusbar.showMessage("Focused", 2000)

        if status == QtMultimedia.QCamera().Unlocked:
            if reason == QtMultimedia.QCamera().LockFailed:
                indicationColor = QtCore.Qt.red
            else:
                indicationColor = QtCore.Qt.black
            self.ui.lockButton.setText("Focus")
            if reason == QtMultimedia.QCamera().LockFailed:
                self.ui.statusbar.showMessage("Focus Failed", 2000)

        palette = QtGui.QPalette(self.ui.lockButton.palette())
        palette.setColor(QtGui.QPalette().ButtonText, indicationColor)
        self.ui.lockButton.setPalette(palette)

    def displayViewfinder(self):
        self.ui.stackedWidget.setCurrentIndex(0)

    def displayCapturedImage(self):
        self.ui.stackedWidget.setCurrentIndex(1)

    def readyForCapture(self, ready):
        self.ui.takeImageButton.setEnabled(ready)

    def imageSaved(self, id, fileName):
        pass

    def keyPressEvent(self, event):
        if event.isAutoRepeat():
            return

        if event.key() == QtCore.Qt.Key_CameraFocus:
            self.displayViewfinder()
            self.camera.searchAndLock()
            event.accept()

        elif event.key() == QtCore.Qt.Key_Camera:
            if self.camera.captureMode() == QtMultimedia.QCamera().CaptureStillImage:
                self.takeImage()
            else:
                if self.mediaRecorder.state() == self.mediaRecorder.RecordingState:
                    self.stop()
                else:
                    self.record()
            event.accept()

        else:
            self.keyPressEvent(event)

    def keyReleaseEvent(self, event):
        if event.isAutoRepeat():
            return

        if event.key() == 0x01100021:
            self.camera.unlock()

        else:
            self.keyReleaseEvent(event)

    def closeEvent(self, event):
        if self.isCapturingImage:
            self.setEnabled(False)
            self.applicationExiting = True
            event.ignore()
        else:
            event.accept()


def main():
    app = QtWidgets.QApplication(sys.argv)
    myapp = Camera()
    myapp.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
