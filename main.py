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

def getTextForRoi(roi):
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
			
	return speedText



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
	ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)

	contours, hier = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
	contours = sorted(contours, key=cv2.contourArea, reverse=True)
	foundContour = False
	for cnt in contours:
		if 200<cv2.contourArea(cnt)<5000:
			x,y,w,h = cv2.boundingRect(cnt)
			#speed signs are relatively round, so ignoring results with too much variation
			if (w/h < 1.2 and h/w < 1.2):
				contour = cnt
				foundContour = True
				#cv2.drawContours(image,[cnt],0,(0,255,0),2)
				#cv2.drawContours(mask,[cnt],0,255,-1)
				break
		
	if (not foundContour):
		return frame

	x,y,w,h = cv2.boundingRect(contour)
	roi = image[y:y+h, x:x+w]

	speedText = getTextForRoi(roi)

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

	centerX = 0
	centerY = 0
	radius = 0
	foundSign = False
	if circles is not None:
		circles = np.uint16(np.around(circles))
		for i in circles[0,:]:
			#cv2.circle(image, (i[0], i[1]), i[2], (0, 255, 0), 2)
			centerX = i[0]
			centerY = i[1]
			radius = i[2]
			foundSign = True
			break
	
	if (not foundSign): # or centerX + radius > image.shape[1] or centerY + radius > image.shape[0]     		or centerX - radius < 0 or centerY - radius < 0
		return image
	x = centerX - radius
	y = centerY - radius
	w = 2*radius
	h = 2*radius

	roi = image[y:y+h, x:x+w]

	if (len(roi) ==0):
		return image

	speedText = getTextForRoi(roi)


	cv2.rectangle(image, [x,y], [x+w, y+h], (255,0,255),4)
	cv2.putText(image, speedText, [x+2,y-5], cv2.FONT_HERSHEY_SIMPLEX, 
                   1, (255,0,255), 2, cv2.LINE_AA)

	"""
	high_thresh, thresh_im = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
	lowThresh = 0.5*high_thresh

	edges = cv2.Canny(gray, lowThresh, high_thresh)
	#circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, 0.8, minDist, param1=param1, param2=param2, minRadius=minRadius, maxRadius=maxRadius)
	circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, 0.8, minDist, param1=param1, param2=param2, minRadius=minRadius, maxRadius=maxRadius)
	

	"""
	return image


def parseSift(image):

	MIN_MATCH_COUNT = 10
	# Initialize SIFT detector
	sift = cv2.SIFT_create(contrastThreshold=0.1)#contrastThreshold=0.1

	comparisonImage = cv2.imread("gimp/30_2.jpg")

	comparisonImage = cv2.resize(comparisonImage, dsize=(160, 160), interpolation=cv2.INTER_CUBIC) #256

	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #
	compGray = cv2.cvtColor(comparisonImage, cv2.COLOR_BGR2GRAY) #

	#gray = cv2.GaussianBlur(gray, (7,7), 0 )
	#compGray = cv2.GaussianBlur(compGray, (7,7), 0 )
	#gray = cv2.resize(gray, (gray.shape[0]*2, gray.shape[1]*2), interpolation = cv2.INTER_AREA)
	"""
	kp = sift.detect(compGray,None)
	#comparisonImage = cv2.drawKeypoints(compGray,kp,comparisonImage)
	kp = sift.detect(comparisonImage,None)
	image = cv2.drawKeypoints(gray,kp,image)
	"""
	signKp, signDesc = sift.detectAndCompute(compGray, None)
	imgKp, imgDesc = sift.detectAndCompute(gray, None)
	

	FLANN_INDEX_KDTREE = 0
	index = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
	search = dict(checks=50)#50
	matcher = cv2.FlannBasedMatcher(index, search)
	matches = matcher.knnMatch(signDesc, imgDesc, k=2)
	
	found = []
	for match in matches:
		if match[0].distance< 0.25 * match[1].distance: #0.5 #original 0.1
			found.append(match)
		#print(match[0].distance,  match[1].distance)
		#print(match[0].trainIdx)


	knn_image = image

	for match in found:
		(x,y) = imgKp[match[0].trainIdx].pt
		x = int(x)
		y = int(y)
		cv2.circle(image, (x, y), 25, (0, 255, 0), 2)


	if (len(found) != 0):
		knn_image = cv2.drawMatchesKnn(comparisonImage, signKp, image, imgKp, matches, None, flags=2)
	return image



#cv2.imshow('result', gray)


files = [f for f in listdir("images/") if isfile(join("images/", f))]
#img = cv2.imread("images/80.jpg")
#fetchSpeedSign(img)

videoFolder = "videos/"
videoName = "Driving in Finland Short Drive in Tampere, Finland.mp4"
"""

cap = cv2.VideoCapture(videoFolder+videoName)
cap.set(cv2.CAP_PROP_POS_FRAMES, 5800)#5200
count = 0
elapsedTime = 0
fps = 10
while cap.isOpened():
	start = time.time()
	ret,frame = cap.read()
	frame = parseSift(frame)
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
	img = parseSift(img)
	cv2.imshow(windowName, img)

	if cv2.waitKey(3000) & 0xFF == ord('q'):
		break
