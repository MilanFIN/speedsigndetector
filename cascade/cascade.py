import numpy as np
import cv2

signCascade = cv2.CascadeClassifier('cascade/cascade.xml')

def detect(image):
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


	signs = signCascade.detectMultiScale(gray, 5, 5)
	
	for (x,y,w,h) in signs:
		cv2.rectangle(image,(x,y),(x+w,y+h),(255,255,0),2)
		break

	return image