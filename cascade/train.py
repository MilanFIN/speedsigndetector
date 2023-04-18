import glob
import subprocess
import os
import time
import sys





glob1 = glob.glob("pos/*")
glob2 = glob.glob("neg/*")
max_xangle, max_yangle, max_zangle = 0.5, 0.5, 0.5
w, h=25,25
#num1 = int(len(glob2) *2/len(glob1)) # 2 means twice as many positive samples as negative images
#num2 = num1+(len(glob2) *2- num1*len(glob1))
num = 3000
infolists = []
for i in range(0, len(glob1)):
                
    #add_prefix_to_filenames("neg/", "sample"+str(i)+"_")
    com="opencv_createsamples -img pos/"+str(i)+".png -bg bg.txt -info info/info"+str(i)+".lst -pngoutput info -maxxangle "+str (max_xangle)+" -maxyangle "+str(max_yangle)+" -maxzangle "+str(max_zangle) +" -num "+str(num) + " -rngseed " + str(i)
    print(com)
    subprocess.call(com, shell=True)
    infolists = infolists+['info/info' +str(i)+'.lst']
    #remove_prefix_from_filenames("neg/", "sample"+str(i)+"_")

store=[]
for j in infolists:
    infolist=open(j, 'r')
    for k in infolist.readlines():
        store=store+[k]
    infolist.close()

final_info=open('info/info.lst', 'w+')
for i in store:
    final_info.write(i)
for j in infolists:
    os.remove(j)


lines = sum(1 for line in open('info/info.lst'))
com2="opencv_createsamples -info info/info.lst -num "+str(lines)+" -w "+str(w)+" -h "+str(h)+" -vec positives.vec"
#len (glob2)
print(com2)
subprocess.call(com2, shell=True)

#finally run:
# opencv_traincascade -data data -vec positives.vec -bg bg.txt -numPos 1800 -numNeg 900 -numStages 10 -w 30 -h 30
# opencv_traincascade -data data -vec positives.vec -bg bg.txt -numPos 89500 -numNeg 9800 -numStages 100 -w 30 -h 30
# opencv_traincascade -data data -vec positives.vec -bg bg.txt -numPos 26928 -numNeg 5000 -numStages 100 -w 25 -h 25
