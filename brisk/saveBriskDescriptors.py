import cv2
import numpy as np
import pickle

brisk = cv2.BRISK_create()

speedSigns = [  "20",
				"30",
				"40",
				"50",
				"60",
				"70",
				"80",
				"100",
				"120"]

with open('briskDescriptors.pkl', 'wb') as output:
	for sign in speedSigns:
		signImg = cv2.imread("sourceImages/"+sign+".jpg")
		signImg = cv2.resize(signImg, dsize=(160, 160), interpolation=cv2.INTER_CUBIC)

		graySign = cv2.cvtColor(signImg, cv2.COLOR_BGR2GRAY)
		signKp, signDesc = brisk.detectAndCompute(graySign, None)

		temp = [{'point0':k.pt[0],'point1':k.pt[1],'size':k.size,'angle': k.angle, 'response': k.response, "octave":k.octave} 
	  				for k in signKp]
		
		print(len(signKp))
		pickle.dump(temp, output)
		pickle.dump(signDesc, output)