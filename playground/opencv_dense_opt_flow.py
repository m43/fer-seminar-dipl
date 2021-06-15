"""
************************************
*** Dense Optical Flow in OpenCV ***
************************************

Tutorial: https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_video/py_lucas_kanade/py_lucas_kanade.html

Lucas-Kanade method computes optical flow for a sparse feature set.
OpenCV provides another algorithm to find the dense optical flow. It computes the optical flow for all the points in the frame. It is based on Gunner Farneback's algorithm
"""

import cv2
import numpy as np
from utils.util import project_path
import os

# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture(os.path.join(project_path, 'data/vtest.avi'))

ret, frame1 = cap.read()
prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
hsv = np.zeros_like(frame1)
hsv[..., 1] = 255

height, width, channels = frame1.shape
video_flow = cv2.VideoWriter(os.path.join(project_path, 'imgs/flow.avi'), cv2.VideoWriter_fourcc(*'MJPG'), 10, (width, height))

while True:
    ret, frame2 = cap.read()
    nxt = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(prvs, nxt, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    hsv[...,0] = ang*180 / np.pi / 2
    hsv[...,2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    cv2.imshow('frame1', frame2)
    cv2.imshow('frame2', rgb)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    elif k == ord('s'):
        cv2.imwrite(os.path.join(project_path,'imgs/opticalfb.png'), frame2)
        cv2.imwrite(os.path.join(project_path,'imgs/opticalhsv.png'), rgb)

    video_flow.write(rgb)
    prvs = nxt

cap.release()
video_flow.release()
cv2.destroyAllWindows()