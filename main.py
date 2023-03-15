import cv2
import numpy as np
import time
from os import listdir
from os.path import isfile, join


channel_initials = list('BGR')


#cv2.imshow('image',image)

def fetchSpeedSign(image):
	ksize = (3, 3)

	result = image.copy()
	output = image.copy()

	#image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

	#image = cv2.blur(image, ksize)
	boundaries = [([0, 0, 150], [150, 100, 255])]
	#[([0, 0, 100], [80, 80, 255])]

	(lower, upper) = boundaries[0]
	lower = np.array(lower, dtype = "uint8")
	upper = np.array(upper, dtype = "uint8")

	mask = cv2.inRange(image, lower, upper)
	result = cv2.bitwise_and(image, image, mask=mask)

	#cv2.imshow('mask', mask)
	gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY) #
	gray = cv2.blur(gray, ksize)

	avg = np.average(gray)
	#converting to black and white
	ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
	print(avg)

	contours, hier = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
	contours = sorted(contours, key=cv2.contourArea, reverse=True)
	for cnt in contours:
		if 200<cv2.contourArea(cnt)<5000:
			contour = cnt
			#cv2.drawContours(image,[cnt],0,(0,255,0),2)
			#cv2.drawContours(mask,[cnt],0,255,-1)
			
			break
		
	x,y,w,h = cv2.boundingRect(contour)
	roi = image[y:y+h, x:x+w]
	cv2.imshow("output", roi)
	cv2.waitKey(3000)


#cv2.imshow('result', gray)


files = [f for f in listdir("images/") if isfile(join("images/", f))]
print(files)
#img = cv2.imread("images/80.jpg")
#fetchSpeedSign(img)

for file in files:
	img = cv2.imread("images/"+file)
	fetchSpeedSign(img)
