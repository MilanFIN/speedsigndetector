from io import StringIO
import csv

speedSigns = [
	"10_SIGN",
	"20_SIGN",
	"30_SIGN",
	"40_SIGN",
	"50_SIGN",
	"60_SIGN",
	"70_SIGN",
	"80_SIGN",
	"90_SIGN",
	"100_SIGN",
	"110_SIGN",
	"120_SIGN",
]

IMAGESIZE = (1280, 960)

with open("./data/annotations.txt", 'r') as file:
	lines = file.readlines()
	for line in lines:
		splitLine = line.split(":")
		image = splitLine[0].split(".")[0]
		speedSign = False
		signCoordinates = []
		#outputLine = image
		outputLines = []

		contents = splitLine[1]
		for signRow in contents.split(";"):
			signContents = signRow.split(",")
			outputLine = ""

			if (len(signContents) > 1):
				signLabel = signContents[-1].strip(" ")
				if (signLabel in speedSigns):
					speedSign = True
					constraints = []
					for i in [3,4,1,2]:
						constraint = round(float(signContents[i]))
						constraints.append(constraint)
					x0 = str(constraints[0])
					y0 = str(constraints[1])
					x1 = str(constraints[2])
					y1 = str(constraints[3])
					signIndex = str(speedSigns.index(signLabel))
					outputLine += signIndex+","+x0+","+y0+","+x1+","+y1 + "\n"
					outputLines.append(outputLine)
					print(outputLine)
		print(image, outputLines)
		with open("./data/annotations/"+image+ ".txt" , "w") as output:
			output.writelines(outputLines)

#for line in outputLines:
#	print(line)
