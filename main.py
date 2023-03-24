import re
import cv2
import numpy as np
import time
from os import listdir
from os.path import isfile, join
import pytesseract


channel_initials = list('BGR')
windowName = "drive"


#cv2.imshow('image',image)

def parseColor(image):
	ksize = (3, 3)

	result = image.copy()
	output = image.copy()

	#image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

	#image = cv2.blur(image, ksize)
	boundaries = [([0, 0, 150], [150, 150, 255])]
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
	foundContour = False
	for cnt in contours:
		if 200<cv2.contourArea(cnt)<5000:
			x,y,w,h = cv2.boundingRect(cnt)
			#speed signs are relatively round, so ignoring results with too much variation
			if (w/h < 1.2 and h/w < 1.2):
				print(w, h)
				contour = cnt
				foundContour = True
				#cv2.drawContours(image,[cnt],0,(0,255,0),2)
				#cv2.drawContours(mask,[cnt],0,255,-1)
				
				break
		
	if (not foundContour):
		return frame

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
	
	kernel = np.zeros((3,3),np.uint8)
	erosion = cv2.erode(finalRoi,kernel,iterations = 1)
	dilate = cv2.dilate(erosion,kernel,iterations = 1)

	rawText = pytesseract.image_to_string(dilate, config="--psm 11 -c tessedit_char_whitelist=0123456789")#
	speedText = ""
	for s in rawText:
		if (s.isdigit()):
			speedText += s
	cv2.rectangle(image, [x,y], [x+w, y+h], (255,0,255),4)
	cv2.putText(image, speedText, [x+2,y-5], cv2.FONT_HERSHEY_SIMPLEX, 
                   1, (255,0,255), 2, cv2.LINE_AA)
	#cv2.imshow('result', image)
	#cv2.waitKey(3000)
	return image

def parseShape(image):

	
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	blur = cv2.medianBlur(gray, 5)

	minDist = 100
	param1 = 50 #500
	param2 = 75 #200 #smaller value-> more false circles
	minRadius = 18
	maxRadius = 100 #10
	circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, 0.8, minDist, param1=param1, param2=param2, minRadius=minRadius, maxRadius=maxRadius)

	if circles is not None:
		circles = np.uint16(np.around(circles))
		for i in circles[0,:]:
			cv2.circle(image, (i[0], i[1]), i[2], (0, 255, 0), 2)
			break



	return image

	"""
	#image = cv2.blur(image, ksize)
	boundaries = [([0, 0, 100], [100, 100, 255])]
	#[([0, 0, 100], [80, 80, 255])]
	(lower, upper) = boundaries[0]
	lower = np.array(lower, dtype = "uint8")
	upper = np.array(upper, dtype = "uint8")

	#mask = cv2.inRange(image, lower, upper)
	#image = cv2.bitwise_and(image, image, mask=mask)
	#image = cv2.blur(image, ksize)

	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #
	
	high_thresh, thresh_im = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+ cv2.THRESH_OTSU ) #+ cv2.THRESH_OTSU
	lowThresh = 0.5*high_thresh

	edges = cv2.Canny(gray,lowThresh,high_thresh)
	edges = cv2.blur(edges, ksize)

	circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, 20, 10, minRadius=20)
	
	if circles is not None:
	# convert the (x, y) coordinates and radius of the circles to integers
		circles = np.round(circles[0, :]).astype("int")
		# loop over the (x, y) coordinates and radius of the circles
		for (x, y, r) in circles:
			# draw the circle in the output image, then draw a rectangle
			# corresponding to the center of the circle
			cv2.circle(image, (x, y), r, (0, 255, 0), 4)
			#cv2.rectangle(image, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
	"""



#cv2.imshow('result', gray)


files = [f for f in listdir("images/") if isfile(join("images/", f))]
#img = cv2.imread("images/80.jpg")
#fetchSpeedSign(img)

videoFolder = "videos/"
videoName = "Driving in Finland Short Drive in Tampere, Finland.mp4"


cap = cv2.VideoCapture(videoFolder+videoName)
cap.set(cv2.CAP_PROP_POS_FRAMES, 5200)#5200
count = 0
elapsedTime = 0
fps = 10
while cap.isOpened():
	start = time.time()
	ret,frame = cap.read()
	frame = parseColor(frame)
	cv2.imshow(windowName, frame)
	count = count + 1
	if cv2.waitKey(10) & 0xFF == ord('q'):
		break
	elapsedTime = (time.time() - start)
	print(count)


cap.release()
cv2.destroyAllWindows() # destroy all opened windows
"""

for file in reversed(files):
	img = cv2.imread("images/"+file)
	#img = parseColor(img)
	img = parseColor(img)
	cv2.imshow(windowName, img)

	if cv2.waitKey(3000) & 0xFF == ord('q'):
		break
"""