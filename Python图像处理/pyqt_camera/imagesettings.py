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
from imagesettings_ui import Ui_ImageSettingsUi

try:
	from PySide import QtWidgets, QtCore, QtGui, QtMultimedia, QtMultimediaWidgets, uic
except:
	try:
		from PyQt5 import QtWidgets, QtCore, QtGui, QtMultimedia, QtMultimediaWidgets, uic
	except:
		six.print_(("Error: can't load PySide or PyQT"), file=sys.stderr, end="\n", sep=" ")
		sys.exit() 

class ImageSettings(QtWidgets.QDialog):
	def __init__(self, imageCapture, parent=None):
		QtWidgets.QDialog.__init__(self, parent)
		# Set up the user interface from Designer.
		#self.ui = uic.loadUi('imagesettings.ui', self)
		self.ui = Ui_ImageSettingsUi()
		self.ui.setupUi(self)
		
		self.imagecapture = imageCapture
		
		#image codecs
		self.ui.imageCodecBox.addItem("Default image format", QtCore.QVariant(six.text_type()))
		for codecName in self.imagecapture.supportedImageCodecs():
			description = self.imagecapture.imageCodecDescription(codecName)
			self.ui.imageCodecBox.addItem(codecName + ": " + description, QtCore.QVariant(codecName))
		
		self.ui.imageQualitySlider.setRange(0, int(QtMultimedia.QMultimedia.VeryHighQuality))
		
		self.ui.imageResolutionBox.addItem("Default Resolution")
		supportedResolutions, check = self.imagecapture.supportedResolutions() # ([], False)
		if check:
			for resolution in supportedResolutions:
				self.ui.imageResolutionBox.addItem(six.text_type("%1x%2") % (((resolution.width(), resolution.height()), QtCore.QVariant(resolution))))

	def changeEvent(self, e):
		QtWidgets.QDialog().changeEvent(e)
		if e.type() == e.LanguageChange:
			self.ui.retranslateUi(self)

	def imageSettings(self):
		settings = self.imagecapture.encodingSettings()
		settings.setCodec(self.boxValue(self.ui.imageCodecBox))
		settings.setQuality(QtMultimedia.QMultimedia.EncodingQuality(self.ui.imageQualitySlider.value()))
		settings.setResolution(self.boxValue(self.ui.imageResolutionBox).toSize())
	
		return settings

	def setImageSettings(self, imageSettings):
		self.selectComboBoxItem(self.ui.imageCodecBox, QtCore.QVariant(imageSettings.codec()))
		self.selectComboBoxItem(self.ui.imageResolutionBox, QtCore.QVariant(imageSettings.resolution()))
		self.ui.imageQualitySlider.setValue(imageSettings.quality())

	def boxValue(self, box):
		idx = box.currentIndex()
		if idx == -1:
			return QtCore.QVariant()
		return box.itemData(idx)

	def selectComboBoxItem(self, box, value):
		for i in range(box.count()):
			if box.itemData(i) == value:
				box.setCurrentIndex(i)
				break