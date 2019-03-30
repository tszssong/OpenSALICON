import numpy as np
from PIL import Image
import pdb
import matplotlib.pyplot as plt
import sys
import time
import scipy.ndimage 
import scipy.misc
sys.path.insert(0,'/nfs/wangshanhu/caffe/python/')
import caffe
caffe.set_mode_gpu()
caffe.set_device(7)
MEAN_VALUE = np.array([103.939, 116.779, 123.68])   # BGR
MEAN_VALUE = MEAN_VALUE[:,None, None]
COARSE_SCALE = np.array([1,3,600,800], dtype=np.float32)

class Salicon:
    def __init__(self, prototxtpath='salicon_mouth.prototxt', model='finetuned_salicon_9789.caffemodel'):
        self.net = caffe.Net(prototxtpath, model, caffe.TEST) 
        
    def process_the_image(self, im):
        # put channel dimension first
        im = np.transpose(im, (2,0,1))
        # switch to BGR
        im = im[::-1, :, :]
        # subtract mean
        im = im - MEAN_VALUE
        im = im[None,:]
        im = im / 255 # convert to float precision
        return im
    
    def compute_saliency(self, image_path):
        im = np.array(Image.open(image_path), dtype=np.float32) # in RGB - ASSUMING 256 IMAGE
        im = self.process_the_image(im)
        coarse_img = scipy.ndimage.interpolation.zoom(im,tuple(COARSE_SCALE / np.asarray(im.shape, dtype=np.float32)), np.dtype(np.float32), mode='nearest')
        assert(coarse_img.shape == (1,3,600,800))
        self.net.blobs['coarse_scale'].data[...] = coarse_img
        self.net.forward()
        sal_map = self.net.blobs['saliency_map_out'].data
        sal_map = sal_map[0,0,:,:]
        sal_map = sal_map - np.amin(sal_map)
        sal_map = sal_map / np.amax(sal_map)    
        sal_map = scipy.ndimage.interpolation.zoom(sal_map,tuple(np.asarray(im.shape[2:], dtype=np.float32) / np.asarray(sal_map.shape,dtype=np.float32)), np.dtype(np.float32), mode='nearest')
        return sal_map

    def compute_saliency_ori(self, image_path):
        im = np.array(Image.open(image_path), dtype=np.float32) # in RGB - ASSUMING 256 IMAGE
        im = self.process_the_image(im)
        coarse_img = scipy.ndimage.interpolation.zoom(im,tuple(COARSE_SCALE / np.asarray(im.shape, dtype=np.float32)), np.dtype(np.float32), mode='nearest')
        assert(coarse_img.shape == (1,3,600,800))
        self.net.blobs['coarse_scale'].data[...] = coarse_img
        self.net.forward()
        sal_map = self.net.blobs['saliency_map_out'].data
        sal_map = sal_map[0,0,:,:]
        sal_map = sal_map - np.amin(sal_map)
        sal_map = sal_map / np.amax(sal_map)    
        return sal_map
