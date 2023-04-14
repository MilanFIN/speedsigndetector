import glob
import subprocess
import os
import time

glob1 = glob.glob("pos/*")
glob2 = glob.glob("neg/*")
max_xangle, max_yangle, max_zangle = 0.5, 0.5, 0.5
w, h=20,20
num1 = int(len(glob2) *2/len(glob1)) # 2 means twice as many positive samples as negative images
num2 = num1+(len(glob2) *2- num1*len(glob1))
infolists = []
for i in range(0, len(glob1)):
	num = num1
	if i==0:
		num = num2
	com="opencv_createsamples -img pos/"+str(i)+".png -bg bg.txt -info info/info"+str(i)+".lst -pngoutput info -maxxangle "+str (max_xangle)+" -maxyangle "+str(max_yangle)+" -maxzangle "+str(max_zangle) +" -num "+str(num)
	subprocess.call(com, shell=True)
	infolists = infolists+['info/info' +str(i)+'.lst']

store=[]
for j in infolists:
	print(infolists)
	infolist=open(j, 'r')
	for k in infolist.readlines():
		store=store+[k]
	infolist.close()

final_info=open('info/info.lst', 'w+')
for i in store:
	final_info.write(i)
for j in infolists:
	os.remove(j)
com2="opencv_createsamples info info/info.lst -num "+str(len (glob2)*2)+" -w "+str(w)+" -h "+str(h)+" -vec positives.vec"
subprocess.call(com2, shell=True)
