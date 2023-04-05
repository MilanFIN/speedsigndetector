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

with open('brisk/briskDescriptors.pkl', 'rb') as inputs:
	for sign in speedSigns:
		rawKeys = pickle.load(inputs)
		keypoints = []
		for kp in rawKeys:
			p = cv2.KeyPoint(x=kp["point0"],y=kp["point0"],size=kp["size"], angle=kp["angle"], response=kp["response"], octave=kp["octave"])
			keypoints.append(p)
		imgKps.append(keypoints)

		desc = pickle.load(inputs)
		imgDescs.append(desc)


def avgKeypointPosition(matches, imgKp):
	if len(matches) == 0:
		return None
	else:
		points = np.zeros((len(matches), 2))
		for i, match in enumerate(matches):
			(x,y) = imgKp[match.trainIdx].pt
			points[i, 0] = x
			points[i, 1] = y
		mean_point = np.mean(points, axis=0)
		return tuple([int(x) for x in mean_point])

def detect(image):

	brisk = cv2.BRISK_create()
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #
	imgKp, imgDesc = brisk.detectAndCompute(gray, None)

	lowestScore = 0
	lowestElem = -1
	bestMatches = []
	for i in range(len(speedSigns)):


		signKp, signDesc = (imgKps[i], imgDescs[i])
		
		matcher = cv2.BFMatcher(normType = cv2.NORM_HAMMING,
							crossCheck = False)
		matches = matcher.match(queryDescriptors = signDesc, trainDescriptors = imgDesc)
				
		matches = sorted(matches, key = lambda x: x.distance)

		if (len(matches) < 10):
			continue

		totalDistance = 0
		if (len(matches) >= 10):
			for match in matches[:10]:
				totalDistance += match.distance **2

			if (lowestElem == -1):
				lowestElem = i
				lowestScore = totalDistance
				bestMatches = matches[:10]
			elif (totalDistance < lowestScore):
				lowestElem = i
				lowestScore = totalDistance
				bestMatches = matches[:10]

		

	if (lowestElem != -1 and lowestScore < 78000):
		"""
		for match in bestMatches:
			(x, y) = imgKp[match.trainIdx].pt
			x = int(x)
			y = int(y)
			cv2.circle(image, (x, y), 3, (255, 0, 255), 2)
		"""
		(x, y) = avgKeypointPosition(bestMatches[:2], imgKp)
		cv2.circle(image, (x, y), 25, (255, 0, 255), 2)


		cv2.putText(image, speedSigns[lowestElem], [x,y-40], cv2.FONT_HERSHEY_SIMPLEX, 
					1, (255,0,255), 2, cv2.LINE_AA)


	return image
