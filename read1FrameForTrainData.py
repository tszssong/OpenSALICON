import sys
import os
import cv2
fromDir = './oriTotal/'
toDir = './Pic/'
corseDir = './coarse_images/'
fineDir = './fine_images/'
if not os.path.isdir(toDir):
    os.makedirs(toDir)
if not os.path.isdir(corseDir):
    os.makedirs(corseDir)
if not os.path.isdir(fineDir):
    os.makedirs(fineDir)
numberVideo = 0
for filename in os.listdir(fromDir):
    if not '.mp4' in filename:                          #skin DS_Store
        continue
    print(filename)
    video = cv2.VideoCapture(fromDir+filename)
    numberVideo += 1;
    success, im = video.read()
    numFrame = 0
    while success:
        if (numFrame%5 == 0) and (not numFrame%2==0):
            savename = filename.split('.')[0]+'_s'+str(numFrame)+'.jpg'
            print savename
            corseImg = cv2.resize(im,(800,600))
            fineImg = cv2.resize(im,(1600,1200))
            cv2.imwrite(toDir+savename, im)
            cv2.imwrite(corseDir+savename, corseImg)
            cv2.imwrite(fineDir+savename, fineImg)
#            break;
        numFrame += 1
        success, im = video.read()
print "total videos:", numberVideo

