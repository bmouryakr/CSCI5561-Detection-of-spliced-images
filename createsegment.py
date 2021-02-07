import os
import xml.etree.ElementTree as ET
from pathlib import Path
import cv2 as cv
from PIL import Image
import numpy as np


darkdir = '/data/VOCdevkit2007/VOC2007/JPEGImages'
darkAnnot = '/data/VOCdevkit2007/VOC2007/Annotations'
darkSeg = '/data/VOCdevkit2007/VOC2007/SegmentationObject'
Path(darkSeg).mkdir(parents=True, exist_ok=True)


for image in os.listdir(darkdir):
    base = os.path.basename(image)
    if os.path.splitext(base)[1]=='.jpg':
        img = cv.imread(os.sep.join([darkdir,image]))
        seg_np = np.zeros(img.shape)
        file = os.path.splitext(base)[0]
        annot = os.sep.join([darkAnnot,file+'.xml'])
        tree = ET.parse(annot)
        counter = 50
        objects=0
        for obj in tree.findall('object'):
            objects+=1
            bbox = obj.find('bndbox')
            xmin = int(bbox.find('xmin').text)
            ymin = int(bbox.find('ymin').text)
            xmax = int(bbox.find('xmax').text)
            ymax = int(bbox.find('ymax').text)
            seg_np[ymin:ymax,xmin:xmax,0]= counter
            counter+=50
            if counter>255:
                counter=counter%255
        cv.imwrite(os.sep.join([darkSeg, file + '.png']),seg_np)