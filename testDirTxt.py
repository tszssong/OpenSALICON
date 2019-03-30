import sys, os
os.environ['GLOG_minloglevel'] = '3'
import cv2
import numpy as np
from SaliconMouth import Salicon
root = "/gpu/zhengmeisong/salient/data/"
f = open(root + sys.argv[1])
sal = Salicon()
for line in f:
    filepath = line.strip()
    videoname = filepath.split('/')[-1].split('.')[0]
    print videoname
    outDir = root + '/' + videoname +'/'
    if not os.path.isdir(outDir):
        os.makedirs(outDir)
    video = cv2.VideoCapture(root + filepath)
    success, im = video.read()
    numFrame = 0
    while success:
        if numFrame%50==0:
          oriname = videoname + '_o' + str(numFrame) + '.jpg'
          salname = videoname + '_s' + str(numFrame) + '.jpg'
          cv2.imwrite(outDir + oriname, im)
          smap = sal.compute_saliency(outDir + oriname)  # open salient need file path
          cv2.imwrite(outDir + salname, smap*255)
          print oriname, salname, "saved in ", outDir
        numFrame += 1
        success, im = video.read()
    

