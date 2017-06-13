"""Denoise input image using pretrained network
Network parameter is specified in cfg['TEST']

Copyright @ Xianming Liu.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
fwd = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(fwd, os.pardir, 'lib'))
os.environ['GLOG_minloglevel'] = '2'
os.environ["MKL_NUM_THREADS"] = "1"  # avoid multithreading

sys.path.append('../caffe/python/')
import caffe
import logging
import numpy as np
from configuration import cfg, cfg_from_file
from util import gray2rgb

__authors__ = ['Xianming Liu (liuxianming@gmail.com)']


Debug = False


def load_network(network_fn, model_fn):
    return caffe.Net(network_fn, model_fn, caffe.TEST)


class ImageDenoiser(object):
    """Class for image denoising
    """
    def __init__(self, cfg_fn):
        cfg_from_file(cfg_fn)
        if cfg['TEST']['DEVICE'] == 'GPU':
            caffe.set_mode_gpu()
            caffe.set_device(int(cfg['TEST']['DEVICE_ID']))
            logging.info("Using GPU {}".format(int(cfg['TEST']['DEVICE_ID'])))
        logging.info(
            "Loading Network {}...".format(cfg['TEST']['NETWORK_NAME']))
        self._net = load_network(cfg['TEST']['NETWORK_FN'],
                                 cfg['TEST']['MODEL_FN'])

    def denoise(self, img):
        """Perform image denoising

        Give:
        img: noisy image as np array
        stride: stride in creating sliding windows
        """
        if img.ndim == 2:
            img = gray2rgb(img)

        # Flip color channel for input image
        img = img[:, :, ::-1]
        img /= float(cfg['SCALE'])

        assert (len(cfg['IMG_MEAN']) == 1 or len(cfg['IMG_MEAN']) == 3), "Format of image mean is NOT correct!"
        if len(cfg['IMG_MEAN']) == 1 and float(cfg['IMG_MEAN'][0]) == -1.0:
            # calculate the mean from image
            cfg['IMG_MEAN'] = np.array([img[:, :, i].mean() for i in range(3)])
            logging.info("Image Mean = {}".format(cfg['IMG_MEAN']))
        else:
            cfg['IMG_MEAN'] = [float(i) for i in cfg['IMG_MEAN']]
        # if need to crop, first expand the image a boarder of size crop
        crop = int(cfg['SAMPLING']['CONV_CROP'])
        if crop > 0:
            exp_img = np.zeros((img.shape[0] + 2 * crop,
                                img.shape[1] + 2 * crop,
                                3)) + cfg['IMG_MEAN']
            exp_img[crop:-crop, crop:-crop, :] = img
            img = exp_img
        self._create_img_blocks(np.copy(img) - cfg['IMG_MEAN'])
        if cfg['FLATTEN']:
            self._blocks = self._blocks.reshape(self._block_count, -1)
        num_batch = int(
            (self._block_count - 1) / cfg['TEST']['MAX_BATCH_SIZE'] + 1)
        denoised_blocks = []
        logging.info("------Start processing blocks------")
        logging.info("Totally {} batches to process".format(num_batch))
        for i in range(num_batch):
            if Debug:
                logging.info("Processing batch {}".format(i))
            idx_start = i * cfg['TEST']['MAX_BATCH_SIZE']
            idx_end = min((i + 1) * cfg['TEST']['MAX_BATCH_SIZE'],
                          self._block_count)
            blocks = self._blocks[idx_start: idx_end, ...]
            self._net.blobs[cfg['TEST']['INPUT_BLOB_NAME']].reshape(
                *(blocks.shape))
            self._net.reshape()
            self._net.blobs[
                cfg['TEST']['INPUT_BLOB_NAME']].data[...] = blocks.astype(
                    np.float, copy=True)
            self._net.forward()
            denoised_blocks.append(
                self._net.blobs[
                    cfg['TEST']['OUTPUT_BLOB_NAME']].data[...].astype(
                        np.float, copy=True))
        if num_batch > 1:
            denoised_blocks = np.vstack(denoised_blocks)
        else:
            denoised_blocks = np.array(denoised_blocks[0])
        denoised_img = self.reconstruct_image(
            denoised_blocks,
            self._img_shape)
        return denoised_img

    def _create_img_blocks(self, img):
        """Create block grids to process
        Stored in self._blocks

        The count of each pixel being covered is stored in self._overlap_count
        """
        logging.info("------Splitting image into blocks------")
        self._img_shape = img.shape
        self._img = img
        crop = int(cfg['SAMPLING']['CONV_CROP'])
        im_h = img.shape[0]
        im_w = img.shape[1]
        if cfg['TEST']['BLOCK_SIZE'][0] == -1 \
           or cfg['TEST']['BLOCK_SIZE'][0] > im_h:
            cfg['TEST']['BLOCK_SIZE'][0] = im_h - crop * 2
        if cfg['TEST']['BLOCK_SIZE'][1] == -1 \
           or cfg['TEST']['BLOCK_SIZE'][1] > im_w:
            cfg['TEST']['BLOCK_SIZE'][1] = im_w - crop * 2

        self._block_grid_x = range(0 + crop,
                                   im_h - cfg['TEST']['BLOCK_SIZE'][0] - crop,
                                   cfg['TEST']['STRIDE'])
        self._block_grid_y = range(0 + crop,
                                   im_w - cfg['TEST']['BLOCK_SIZE'][1] - crop,
                                   cfg['TEST']['STRIDE'])
        self._block_grid_x += [im_h - cfg['TEST']['BLOCK_SIZE'][0] - crop]
        self._block_grid_y += [im_w - cfg['TEST']['BLOCK_SIZE'][1] - crop]

        self._block_grid_dims = [len(self._block_grid_x),
                                 len(self._block_grid_y)]
        self._block_count = np.prod(np.array(self._block_grid_dims))

        self._overlap_count = np.zeros(img.shape[:2])
        _blocks = []
        for i in range(self._block_grid_dims[0]):
            for j in range(self._block_grid_dims[1]):
                start_x = self._block_grid_x[i]
                start_y = self._block_grid_y[j]
                _block = img[
                    start_x - crop:
                    start_x + cfg['TEST']['BLOCK_SIZE'][0] + crop,
                    start_y - crop:
                    start_y + cfg['TEST']['BLOCK_SIZE'][1] + crop,
                    :].astype(np.float, copy=True)
                self._overlap_count[
                    start_x: start_x + cfg['TEST']['BLOCK_SIZE'][0],
                    start_y: start_y + cfg['TEST']['BLOCK_SIZE'][1]] += 1
                _blocks.append(_block.transpose(2, 0, 1))
        self._blocks = np.array(_blocks)
        logging.info("Splitting completed. {} blocks".format(self._block_count))

    def reconstruct_image(self, output_blocks, img_shape):
        """Reconstruct denoised image
        """
        denoised_img = np.zeros(img_shape)
        block_size = cfg['TEST']['BLOCK_SIZE']
        logging.info(output_blocks.shape)
        for i in range(self._block_grid_dims[0]):
            for j in range(self._block_grid_dims[1]):
                idx = i * self._block_grid_dims[1] + j
                block_ = output_blocks[idx, ...]
                if cfg['FLATTEN']:
                    block_ = block_.reshape(
                        -1, block_size[0], block_size[1])
                block_ = block_.transpose(1, 2, 0)
                start_x = self._block_grid_x[i]
                start_y = self._block_grid_y[j]
                denoised_img[start_x: start_x + block_size[0],
                             start_y: start_y + block_size[1],
                             :] += block_
        denoised_img /= self._overlap_count[:, :, np.newaxis] + np.spacing(1)
        denoised_img += cfg['IMG_MEAN']
        crop = int(cfg['SAMPLING']['CONV_CROP'])
        if crop > 0:
            denoised_img = denoised_img[crop:-crop, crop:-crop, :]

        # scale back to [0, 255]
        denoised_img *= float(cfg['SCALE'])
        # flip color channel back
        denoised_img = denoised_img[:, :, ::-1]
        # clip image to fall within [0, 255] and round to integer
        denoised_img = np.clip(np.rint(denoised_img), 0, 255.0)
        return denoised_img
