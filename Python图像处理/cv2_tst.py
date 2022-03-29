# -*- coding: utf-8 -*-
"""
Created on Fri Feb 18 20:18:05 2022

@author: lei
"""

import cv2

camera = cv2.VideoCapture(0)  # \@视频\豪华游轮(上).mp4")
if camera.isOpened():
    print(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
    f, img = camera.read()
    cv2.imwrite("cap1.jpg", img)
else:
    print("no camera opened")
cv2.destroyAllWindows()
