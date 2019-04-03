import sys, os
os.environ['GLOG_minloglevel'] = '3'
import cv2
import numpy as np

def calc_cc_score(gtsAnn, resAnn):
    """
        Computer CC score. A simple implementation
        :param gtsAnn : ground-truth fixation map
        :param resAnn : predicted saliency map
        :return score: int : score
        """
    fixationMap = gtsAnn - np.mean(gtsAnn)
    if np.max(fixationMap) > 0:
        fixationMap = fixationMap / np.std(fixationMap)
    salMap = resAnn - np.mean(resAnn)
    if np.max(salMap) > 0:
        salMap = salMap / np.std(salMap)
    
    return np.corrcoef(salMap.reshape(-1), fixationMap.reshape(-1))[0][1]

def calc_kl_score(gtsAnn, resAnn, eps = 1e-7):
    gtsAnn.astype(np.float32)
    resAnn.astype(np.float32)
    if np.sum(gtsAnn) > 0:
        gtsAnn = gtsAnn / float(np.sum(gtsAnn))
    if np.sum(resAnn) > 0:
        resAnn = resAnn / float(np.sum(resAnn))
    return np.sum(gtsAnn * np.log(eps + gtsAnn / (resAnn + eps)))

from SaliconRes import Salicon
sal = Salicon()
fromDir = '/gpu/zhengmeisong/salient/testData/'
toDir_small = '/gpu/zhengmeisong/salient/testData/retSmall/'
if not os.path.isdir(toDir_small):
    os.makedirs(toDir_small)
SAVE_PIC = False
filelist = open(fromDir + '/test.txt')
cc_list = []
kl_list = []
for line in filelist.readlines():
    imgname = line.strip()
    map = sal.compute_saliency_ori(fromDir + 'coarse_images/' + imgname)
    gt = cv2.imread(gtsDir + imgname)
    kl = calc_kl_score(gt, re)
    kl_list.append(kl)
    cc = calc_cc_score(gt, re)
    cc_list.append(cc)
    print "%15s"%(imgname), "%.2f"%(cc), "%.2f"%(kl)
    if SAVE_PIC:
        cv2.imwrite(toDir_small + imgname, map*255)
print "ave_cc = %.3f, ave_kl = %.3f"%( np.mean(cc_list), np.mean(kl_list) )
