import sys, os
os.environ['GLOG_minloglevel'] = '3'
import cv2
from SaliconRes import Salicon
sal = Salicon()
fromDir = '/gpu/zhengmeisong/salient/testData/'
toDir = '/gpu/zhengmeisong/salient/testData/ret/'
toDir_small = '/gpu/zhengmeisong/salient/testData/retSmall/'
if not os.path.isdir(toDir):
    os.makedirs(toDir)
if not os.path.isdir(toDir_small):
    os.makedirs(toDir_small)
filelist = open(fromDir + '/test.txt')
for line in filelist.readlines():
    imgname = line.strip()
    map = sal.compute_saliency(fromDir + 'coarse_images/' + imgname)
    print imgname, map.shape
    cv2.imwrite(toDir + imgname, map*255)
    map = sal.compute_saliency_ori(fromDir + 'coarse_images/' + imgname)
    print imgname, map.shape
    cv2.imwrite(toDir_small + imgname, map*255)
