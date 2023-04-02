import re
import cv2
import numpy as np
import time
from os import listdir
from os.path import isfile, join
import pytesseract

from redChannel import redChannel
from circle import circle
from sift import sift
from brisk import brisk


channel_initials = list('BGR')
windowName = "drive"


#cv2.imshow('image',image)






#cv2.imshow('result', gray)


files = [f for f in listdir("images/") if isfile(join("images/", f))]
#img = cv2.imread("images/80.jpg")
#fetchSpeedSign(img)

videoFolder = "videos/"
videoName = "Driving in Finland Short Drive in Tampere, Finland.mp4"

"""

cap = cv2.VideoCapture(videoFolder+videoName)
cap.set(cv2.CAP_PROP_POS_FRAMES, 5200)#5200
count = 0
elapsedTime = 0
fps = 10
while cap.isOpened():
	start = time.time()
	ret,frame = cap.read()
	frame = sift.detect(frame)
	cv2.imshow(windowName, frame)
	count = count + 1
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
	elapsedTime = (time.time() - start)
	print(count)


cap.release()
cv2.destroyAllWindows() # destroy all opened windows
"""

for file in reversed(files):
	img = cv2.imread("images/"+file)
	#img = parseColor(img)
	img = brisk.detect(img)
	cv2.imshow(windowName, img)

	if cv2.waitKey(3000) & 0xFF == ord('q'):
		break
