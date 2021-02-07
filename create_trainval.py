import os
a = open("/data/DIY_dataset/VOC2007/ImageSets/Main/trainval.txt", "w")
for path, subdirs, files in os.walk(r'/data/DIY_dataset/VOC2007/Annotations'):
    for filename in files:
        f1=str(filename).replace(".xml","")
        #print(f1)
        #f = os.path.join(path, filename)
        a.write(str(f1)+os.linesep)
a.close()
