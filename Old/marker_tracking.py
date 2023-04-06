# -*- coding: utf-8 -*-
"""
"""

import time, io, os, glob, pickle, socket, threading, subprocess
#import paramiko
import picamera
import cv2
import matplotlib.pylab as plt
import pandas as pd
import numpy as np
from natsort import natsort
import pdb, itertools



images = glob.glob('*npy')
images = natsort.humansorted(images)
im = np.load(images[0])



params = cv2.SimpleBlobDetector_Params()

# Change thresholds
params.minThreshold = 0
params.maxThreshold = 255

params.filterByColor = True
params.blobColor = 255

params.filterByCircularity = True
params.minCircularity = 0.8





ver = (cv2.__version__).split('.')
if int(ver[0]) < 3 :
	detector = cv2.SimpleBlobDetector(params)
else : 
	detector = cv2.SimpleBlobDetector_create(params)


# Detect blobs.
keypoints = detector.detect(im)

# Draw detected blobs as red circles.
# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures
# the size of the circle corresponds to the size of blob

im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Show blobs
cv2.imshow("Keypoints", im_with_keypoints)
cv2.waitKey(0)