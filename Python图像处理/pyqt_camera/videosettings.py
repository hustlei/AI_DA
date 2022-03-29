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
** "Redistribution and use in source and binary forms, with or without
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
from videosettings_ui import Ui_VideoSettingsUi

try:
	from PySide import QtWidgets, QtCore, QtGui, QtMultimedia, QtMultimediaWidgets, uic
except:
	try:
		from PyQt5 import QtWidgets, QtCore, QtGui, QtMultimedia, QtMultimediaWidgets, uic
	except:
		six.print_(("Error: can't load PySide or PyQT"), file=sys.stderr, end="\n", sep=" ")
		sys.exit() 

class VideoSettings(QtWidgets.QDialog):
	def __init__(self, mediaRecorder, parent=None):
		QtWidgets.QDialog.__init__(self, parent)
		# Set up the user interface from Designer.
		#self.ui = uic.loadUi('videosettings.ui', self)
		self.ui = Ui_VideoSettingsUi()
		self.ui.setupUi(self)
		
		self.mediaRecorder = mediaRecorder
		
		self.audioSettings = QtMultimedia.QAudioEncoderSettings()
		self.videoSettings = QtMultimedia.QVideoEncoderSettings()
		
		#audio codecs
		self.ui.audioCodecBox.addItem("Default audio codec", QtCore.QVariant(six.text_type()))
		for codecName in self.mediaRecorder.supportedAudioCodecs():
			 description = self.mediaRecorder.audioCodecDescription(codecName)
			 self.ui.audioCodecBox.addItem(codecName + ": " + description, QtCore.QVariant(codecName))
		 
		#sample rate:
		for sampleRate in self.mediaRecorder.supportedAudioSampleRates():
			self.ui.audioSampleRateBox.addItem(six.text_type(sampleRate), QtCore.QVariant(sampleRate))
		
		self.ui.audioQualitySlider.setRange(0, QtMultimedia.QMultimedia.VeryHighQuality)
		
		#video codecs
		self.ui.videoCodecBox.addItem("Default video codec", QtCore.QVariant(six.text_type()))
		for codecName in self.mediaRecorder.supportedVideoCodecs():
			description = self.mediaRecorder.videoCodecDescription(codecName)
			self.ui.videoCodecBox.addItem(codecName + ": " + description, QtCore.QVariant(codecName))
		
		self.ui.videoQualitySlider.setRange(0, QtMultimedia.QMultimedia.VeryHighQuality)
		
		self.ui.videoResolutionBox.addItem("Default")
		supportedResolutions, check = self.mediaRecorder.supportedResolutions() # ([], False)
		if check:
			for resolution in supportedResolutions:
				self.ui.videoResolutionBox.addItem(six.text_type("%1x%2") % (((resolution.width(),resolution.height()), QtCore.QVariant(resolution))))
		
		self.ui.videoFramerateBox.addItem("Default")
		supportedFrameRates, check = self.mediaRecorder.supportedFrameRates()
		if check:
			for rate in supportedFrameRates:
				rateString = six.text_type("%0.2f") % ((rate))
				self.ui.videoFramerateBox.addItem(rateString, QtCore.QVariant(rate))
		
		#containers
		self.ui.containerFormatBox.addItem("Default container", QtCore.QVariant(six.text_type()))
		for format in self.mediaRecorder.supportedContainers():
			self.ui.containerFormatBox.addItem(format + ":" + mediaRecorder.containerDescription(format), QtCore.QVariant(format))

	def setAudioSettings(self, audioSettings):
		self.selectComboBoxItem(self.ui.audioCodecBox, QtCore.QVariant(audioSettings.codec()))
		self.selectComboBoxItem(self.ui.audioSampleRateBox, QtCore.QVariant(audioSettings.sampleRate()))
		self.ui.audioQualitySlider.setValue(audioSettings.quality())

	def audioSettings(self):
		settings = self.mediaRecorder.audioSettings()
		settings.setCodec(self.boxValue(self.ui.audioCodecBox).toString())
		settings.setQuality(QtMultimedia.QMultimedia.EncodingQuality(self.ui.audioQualitySlider.value()))
		settings.setSampleRate(self.boxValue(self.ui.audioSampleRateBox).toInt())
		return settings

	def videoSettings(self):
		settings = self.mediaRecorder.videoSettings()
		settings.setCodec(self.boxValue(self.ui.videoCodecBox).toString())
		settings.setQuality(QtMultimedia.QMultimedia.EncodingQuality(self.ui.videoQualitySlider.value()))
		settings.setResolution(self.boxValue(self.ui.videoResolutionBox).toSize())
		settings.setFrameRate(float(self.boxValue(self.ui.videoFramerateBox)))

		return settings

	def setVideoSettings(self, videoSettings):
		self.selectComboBoxItem(self.ui.videoCodecBox, QtCore.QVariant(self.videoSettings.codec()))
		self.selectComboBoxItem(self.ui.videoResolutionBox, QtCore.QVariant(self.videoSettings.resolution()))
		self.ui.videoQualitySlider.setValue(self.videoSettings.quality())

		#special case for frame rate
		for i in range(self.ui.videoFramerateBox.count()):
			if not self.ui.videoFramerateBox.itemData(i): continue
			itemRate = float(self.ui.videoFramerateBox.itemData(i))
			if QtCore.QtGlobal.qFuzzyCompare(itemRate, videoSettings.frameRate()):
				self.ui.videoFramerateBox.setCurrentIndex(i)
				break

	def format(self):
		return self.boxValue(self.ui.containerFormatBox)

	def setFormat(self, format):
		self.selectComboBoxItem(self.ui.containerFormatBox, QtCore.QVariant(format))

	def boxValue(self, box):
		idx = box.currentIndex()
		if idx == -1:
			return QtCore.QVariant()
		
		return box.itemData(idx)

	def changeEvent(e):
		QtWidgets.QDialog().changeEvent(e)
		if e.type() == e.LanguageChange:
			self.ui.retranslateUi(self)

	def selectComboBoxItem(self, box, value):
		for i in range(box.count()):
			if box.itemData(i) == value:
				box.setCurrentIndex(i)
				break
