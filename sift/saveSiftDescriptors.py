import cv2
import numpy as np
import pickle

sift = cv2.SIFT_create(contrastThreshold=0.1)#contrastThreshold=0.1

speedSigns = [  "20",
				"30",
				"40",
				"50",
				"60",
				"70",
				"80",
				"100",
				"120"]

with open('siftDescriptors.pkl', 'wb') as output:
	for sign in speedSigns:
		signImg = cv2.imread("sourceImages/"+sign+".jpg")
		graySign = cv2.cvtColor(signImg, cv2.COLOR_BGR2GRAY) #
		signKp, signDesc = sift.detectAndCompute(graySign, None)

		#print(signKp[0].size)
		
		temp = [{'point0':k.pt[0],'point1':k.pt[1],'size':k.size,'angle': k.angle, 'response': k.response, "octave":k.octave} 
	  				for k in signKp]
		
		print(len(signKp))
		pickle.dump(temp, output)
		pickle.dump(signDesc, output)
		

