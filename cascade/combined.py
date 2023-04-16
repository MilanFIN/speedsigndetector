import glob
import subprocess
import os
import time




def add_prefix_to_filenames(folder_path, prefix):
    # List all files in the folder
    files = os.listdir(folder_path)
    
    # Rename only .jpg files in the folder
    for file in files:
        if file.endswith('.jpg'):
            # Rename the file with the prefix
            new_file = prefix + file
            os.rename(os.path.join(folder_path, file), os.path.join(folder_path, new_file))
    
    # Modify the file names in bg.txt to correspond to the added prefixes
    with open('bg.txt', 'r') as f:
        lines = f.readlines()
        
    with open('bg.txt', 'w') as f:
        for line in lines:
            if line.startswith('neg/'):
                # Extract the filename from the line
                filename = line.split('/')[-1].strip()
                
                # Add the prefix to the filename
                new_filename = prefix + filename
                
                # Replace the old filename with the new one
                new_line = line.replace(filename, new_filename)
                f.write(new_line)
            else:
                f.write(line)

def remove_prefix_from_filenames(folder_path, prefix):
    # List all files in the folder
    files = os.listdir(folder_path)
    
    # Revert only .jpg files in the folder
    for file in files:
        if file.endswith('.jpg') and file.startswith(prefix):
            # Revert the file name to remove the prefix
            new_file = file[len(prefix):]
            os.rename(os.path.join(folder_path, file), os.path.join(folder_path, new_file))
    
    # Modify the file names in bg.txt to revert the filename changes back to original
    with open('bg.txt', 'r') as f:
        lines = f.readlines()
        
    with open('bg.txt', 'w') as f:
        for line in lines:
            if line.startswith('neg/'):
                # Extract the filename from the line
                filename = line.split('/')[-1].strip()
                
                if filename.startswith(prefix):
                    # Revert the file name to remove the prefix
                    new_filename = filename[len(prefix):]
                else:
                    new_filename = filename
                
                # Replace the old filename with the new one
                new_line = line.replace(filename, new_filename)
                f.write(new_line)
            else:
                f.write(line)




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


com2="opencv_createsamples -info info/info.lst -num "+str(len (glob2))+" -w "+str(w)+" -h "+str(h)+" -vec positives.vec"

print(com2)
subprocess.call(com2, shell=True)

#finally run:
# opencv_traincascade -data data -vec positives.vec -bg bg.txt -numPos 1800 -numNeg 900 -numStages 10 -w 20 -h 20
