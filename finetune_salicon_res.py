import numpy as np
from PIL import Image
import pdb
import matplotlib.pyplot as plt
import sys
import time
sys.path.insert(0, '/nfs/wangshanhu/caffe/python') # UPDATE YOUR CAFFE PATH HERE
#sys.path.insert(0, '/nfs/zhengmeisong/wkspace/caffe/python') # UPDATE YOUR CAFFE PATH HERE
import caffe
caffe.set_mode_gpu()
caffe.set_device(7)
fine_imgs = []
coarse_imgs = []
fix_imgs = []
training_data_path = '/gpu/zhengmeisong/salient/salData/' # PATH TO YOUR TRAINING DATA
MEAN_VALUE = np.array([103.939, 116.779, 123.68])   # BGR
MEAN_VALUE = MEAN_VALUE[:,None, None]
listpath = '/gpu/zhengmeisong/salient/salData/buaa.txt'
idx = 0
for line in open(listpath):
    imgname = line.strip()
    im = np.array(Image.open(training_data_path + 'coarse_images/' + imgname), dtype=np.float32) # in RGB
    # put channel dimension first
    im = np.transpose(im, (2,0,1))
    # switch to BGR
    im = im[::-1, :, :]
    # subtract mean
    im = im - MEAN_VALUE
    im = im[None,:]
    assert(im.shape == (1,3,600,800))
    im = im / 255
    im = im.astype(np.dtype(np.float32))
    coarse_imgs.append(im)
    idx += 1
    if(idx%500==0):
        print idx, "images read"
idx = 0
for line in open(listpath):
    imgname = line.strip()
    im = np.array(Image.open(training_data_path + 'fixation_images/' + imgname), dtype=np.float32)
    im = np.transpose(im, (2,0,1))
    im = im[0]
    im = im[None,None,:]
    assert(im.shape == (1,1,38,50))
    im = im / 255
    im = im.astype(np.dtype(np.float32))
    fix_imgs.append(im)
    idx += 1
    if(idx%500==0):
        print idx, "images read"

print len(coarse_imgs), len(fix_imgs)
assert(len(fix_imgs) == len(coarse_imgs))
#assert(len(fix_imgs) == 700)
solver = caffe.SGDSolver('solver_new.prototxt')
solver.net.copy_from('res18_salicon_95849.caffemodel') # untrained.caffemodel
start_time = time.time()
idx_counter = 0
while time.time() - start_time < 432000:
    batch = np.random.permutation(len(fix_imgs))
    for i in range(0, len(batch)):
        idx_counter = idx_counter + 1
        coarse_img_to_process = coarse_imgs[batch[i]]
        fix_img_to_process = fix_imgs[batch[i]]
        solver.net.blobs['coarse_scale'].data[...] = coarse_img_to_process
        solver.net.blobs['ground_truth'].data[...] = fix_img_to_process
        solver.step(1)
        if int(time.time() - start_time) % 10000 == 0:
            solver.net.save('train_output/fintune2_res18_salicon_{}.caffemodel'.format(idx_counter))
    solver.net.save('train_output/fintune2_res18_salicon_{}.caffemodel'.format(idx_counter))
