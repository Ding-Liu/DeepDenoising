"""Util.py

Util functions for deepdenoising,
including I/O functions
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import scipy.misc
import numpy as np


def load_image(img_fn):
    img = scipy.misc.imread(img_fn)  # read image as RGB
    img = img.astype(np.float)
    # img /= 255
    return img


def rgb2ycbcr(im_rgb):
    im_rgb = im_rgb.astype(np.float32)
    im_ycrcb = cv2.cvtColor(im_rgb, cv2.COLOR_RGB2YCR_CB)
    im_ycbcr = im_ycrcb[:, :, (0, 2, 1)].astype(np.float32)
    # to [16/255, 235/255]
    im_ycbcr[:, :, 0] = (im_ycbcr[:, :, 0] * (235 - 16) + 16) / 255.0
    # to [16/255, 240/255]
    im_ycbcr[:, :, 1:] = (im_ycbcr[:, :, 1:] * (240 - 16) + 16) / 255.0
    return im_ycbcr


def gray2rgb(img_gray):
    img_gray = img_gray.astype(np.float32)
    img_rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
    return img_rgb.astype(np.float)


def computePSNR(im1, im2, cfg, ColorChannel=False):
    """
    im1: float np array in [0, 255]:
    im2: float np array in [0, 255]:
    ColorChannel:
        True: PSNR calculated over RGB channels
        False: PSNR calculated only on luma channel after colorspace conversion
    """
    crop = int(cfg['SAMPLING']['CONV_CROP'])
    if crop > 0:
        im1 = im1[crop: -crop, crop: -crop]
        im2 = im2[crop: -crop, crop: -crop]
    # im1 = np.clip(im1, 0, 1)
    # im2 = np.clip(im2, 0, 1)
    if len(im1.shape) == 3 and im1.shape[2] == 3 and ColorChannel is False:
        im1 = rgb2ycbcr(im1/255.0)[:, :, 0]*255.0
        im2 = rgb2ycbcr(im2/255.0)[:, :, 0]*255.0

    im1_uint8 = np.rint(np.clip(im1, 0, 255))
    im2_uint8 = np.rint(np.clip(im2, 0, 255))

    diff = np.abs(im1_uint8 - im2_uint8).flatten()
    rmse = np.sqrt(np.mean(np.square(diff)))
    psnr = 20 * np.log10(255.0 / rmse)
    return rmse, psnr
