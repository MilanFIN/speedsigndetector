import cv2
import numpy as np
import pytesseract


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



def detect(image):

	
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