

from os import listdir
from os.path import isfile, join

from PIL import Image

files = [f for f in listdir("pos/") if isfile(join("pos/", f))]


i = 0
for file in files:
    img = Image.open('pos/'+file)
    img = img.convert("RGBA")
    img = img.convert('LA')
    datas = img.getdata()

    newData = []
    for item in datas:
        if item[0] > 240 and item[1] == 255:# and item[1] == 255 and item[2] == 255
            newData.append((255, 0))
        else:
            newData.append(item)

    img.putdata(newData)
    img = img.resize((50, 50))
    img.save("pos/" + str(i) + ".png", "PNG")
    i += 1
