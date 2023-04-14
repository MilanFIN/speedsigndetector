
from os import listdir
from os.path import isfile, join
import cv2 
import numpy as np


def fetchNegatives():
	src_dir = "images/hug/"
	dst_dir = "neg/"

	files = [f for f in listdir("images/hug/") if isfile(join("images/hug/", f))]

	i = 0

	for file in files:
		print(src_dir+file)
		try:
			img = cv2.imread(src_dir+file)
			gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 

			gray = cv2.resize(gray, dsize=(100, 100), interpolation=cv2.INTER_CUBIC)
			cv2.imwrite(dst_dir + str(i) + ".jpg", gray)
			i += 1
		except Exception:
			pass
		if i >= 2000:
			break


def createNegativePos():

	
	for img in listdir("neg"):

		line = 'neg'+'/'+img+'\n'
		with open('bg.txt','a') as f:
			f.write(line)



#fetchNegatives()
#createNegativePos()
