import cv2
import numpy as np
import time
import math
from djitellopy import Tello
from pyimagesearch.pid import PID


# Load the predefined dictionary
dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)

# Initialize the detector parameters using default values
parameters = cv2.aruco.DetectorParameters_create()

# Detect the markers in the image
markerCorners, markerIds, rejectedCandidates = \
cv2.aruco.detectMarkers(frame, dictionary, parameters=parameters)

frame = cv2.aruco.drawDetectedMarkers(frame, markerCorners, markerIds)

#Pose estimation for single markers.
rvec, tvec, _objPoints = \
cv2.aruco.estimatePoseSingleMarkers(markerCorners, 15, intrinsic, distortion)

frame = cv2.aruco.drawAxis(frame, intrinsic, distortion, rvec, tvec, 0.1)

# Get the parameters of camera calibration
fs = cv2.FileStorage("calibrateCamera.xml", cv2.FILE_STORAGE_READ)
intrinsic = fs.getNode("intrinsic").mat()
distortion = fs.getNode('distortion').mat()

