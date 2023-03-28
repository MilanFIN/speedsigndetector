import cv2
import numpy as np
import pickle
import json


speedSigns = [  "20",
				"30",
				"40",
				"50",
				"60",
				"70",
				"80",
				"100",
				"120"]

imgKps = []
imgDescs = []

with open('siftDescriptors.pkl', 'rb') as inputs:
	for sign in speedSigns:
		rawKeys = pickle.load(inputs)
		print(len(rawKeys))
		keypoints = []
		for kp in rawKeys:
			p = cv2.KeyPoint(x=kp["point0"],y=kp["point0"],size=kp["size"], angle=kp["angle"], response=kp["response"], octave=kp["octave"])
			keypoints.append(p)
		imgKps.append(keypoints)

		desc = pickle.load(inputs)
		imgDescs.append(desc)
