import sys, os
os.environ['GLOG_minloglevel'] = '3'
import cv2
import numpy as np
from Salicon import Salicon
root = "/gpu/zhengmeisong/salient/data/"
f = open(sys.argv[1])
outDir = root + 's255_out/'
if not os.path.isdir(outDir):
    os.makedirs(outDir)
for line in f:
    filepath = line.strip()
    filename = filepath.split('/')[-1]
    print filepath,filename
    sal = Salicon()
    map = sal.compute_saliency(root+filepath)
    cv2.imwrite(outDir+filename, map*255)
    np.savetxt(outDir+filename.split('.')[0]+'.txt', map*255, fmt="%d")
    

