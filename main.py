import cv2
import numpy as np
import time
from os import listdir
from os.path import isfile, join
import pytesseract


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

	redRoi = roi.copy()
	redRoi[:,:,0] = np.zeros([redRoi.shape[0], redRoi.shape[1]])
	redRoi[:,:,1] = np.zeros([redRoi.shape[0], redRoi.shape[1]])

	grayRoi = cv2.cvtColor(redRoi, cv2.COLOR_BGR2GRAY) 
	roiMax = np.max(grayRoi)

	ret, threshedRoi = cv2.threshold(grayRoi, roiMax *0.5, 255, cv2.THRESH_BINARY_INV)

	roiKSize = (3, 3)

	blurredRoi = cv2.blur(threshedRoi, roiKSize)

	inverseGrayRoi = cv2.bitwise_not(grayRoi)
	finalRoi = cv2.bitwise_and(inverseGrayRoi, inverseGrayRoi, mask=blurredRoi)
	inverseFinalRoi = cv2.bitwise_not(finalRoi)
	
	kernel = np.zeros((3,3),np.uint8)
	erosion = cv2.erode(finalRoi,kernel,iterations = 1)
	dilate = cv2.dilate(erosion,kernel,iterations = 1)

	text = pytesseract.image_to_string(dilate, config="--psm 11 -c tessedit_char_whitelist=0123456789")#
	print(text)
	cv2.imshow('result', dilate)
	cv2.waitKey(3000)



	custom_config = r'--oem 3 --psm 11 -c tessedit_char_whitelist=0123456789'



#cv2.imshow('result', gray)


files = [f for f in listdir("images/") if isfile(join("images/", f))]
#img = cv2.imread("images/80.jpg")
#fetchSpeedSign(img)

for file in files:
	img = cv2.imread("images/"+file)
	fetchSpeedSign(img)
