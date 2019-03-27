import sys, os
os.environ['GLOG_minloglevel'] = '3'
import cv2
import numpy as np
from Salicon import Salicon
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
        oriname = videoname + '_o' + str(numFrame) + '.jpg'
        salname = videoname + '_s' + str(numFrame) + '.jpg'
        txtname = videoname + '_' + str(numFrame) + '.txt'
        numFrame += 1
        cv2.imwrite(outDir + oriname, im)
        smap = sal.compute_saliency_ori(outDir + oriname)  # open salient need file path
        img = cv2.resize(im, (smap.shape[1],smap.shape[0])) #caffe NCHW, Opencv resize(Width, Height)
        print smap.shape, im.shape, img.shape
        cv2.imwrite(outDir + oriname, img)
        cv2.imwrite(outDir + salname, smap*255)
        np.savetxt(outDir + txtname, smap*255, fmt="%d")
        print oriname, salname, txtname, "saved in ", outDir
        success, im = video.read()
    

