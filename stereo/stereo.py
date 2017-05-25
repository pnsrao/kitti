#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon May 15 19:02:53 2017

@author: subbu
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import os
import glob
import cv2


# Change this to the directory where you store KITTI data
basedir = 'C:/Users/home/Documents/GitHub/kitti/raw_data'

# Specify the dataset to load
date = '2011_09_26'
drive = '0017'

drive_str = date + '_drive_' + drive + '_sync'
imagedir = os.path.join(basedir,date,drive_str)

print('Loading monochrome images from ' + drive_str + '...')

imL_path = os.path.join(imagedir, 'image_00', 'data', '*.png')
imR_path = os.path.join(imagedir, 'image_01', 'data', '*.png')

imL_files = sorted(glob.glob(imL_path))
imR_files = sorted(glob.glob(imR_path))

num_frames = len(imL_files)
assert len(imR_files)==num_frames, "Number of left and right frames don't match"

#imL = []
#imR = []
#for ii in range(num_frames):
#    imL.append(mpimg.imread(imL_files[ii]))
#    imR.append(mpimg.imread(imR_files[ii]))

#print('Loading color images from ' + drive_str + '...')
#
#imL_path = os.path.join(imagedir, 'image_02', 'data', '*.png')
#imR_path = os.path.join(imagedir, 'image_03', 'data', '*.png')
#
#imL_files = sorted(glob.glob(imL_path))
#imR_files = sorted(glob.glob(imR_path))

#f, ax = plt.subplots(2, 1, figsize=(30, 10))
#
#ax[0].imshow(imL[0],cmap='gray')
#lpts = ax[0].ginput(3)
#ax[1].imshow(imR[0],cmap='gray')
#rpts = ax[0].ginput(3)


imL = cv2.imread(imL_files[0],0)
imR = cv2.imread(imR_files[0],0)

min_disp=16
num_disp=112-min_disp
#stereo = cv2.createStereoBM(numDisparities=16, blockSize=15)
stereo = cv2.StereoSGBM_create(minDisparity = min_disp, numDisparities=num_disp, blockSize=15)
disp = stereo.compute(imL,imR).astype(np.float32) / 16.0
plt.imshow(disp, 'gray')
#plt.show()
#pts = plt.ginput(3)
#print pts
cv2.imshow('left', imL)
cv2.imshow('disparity', (disp-min_disp)/num_disp)
cv2.waitKey()
cv2.destroyAllWindows()