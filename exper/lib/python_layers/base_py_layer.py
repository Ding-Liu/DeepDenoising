"""Base python layers
1. PrefetchDataLayer - python layer with prefetching
2. BaseImageVariationLayer - Base layer for image variations
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import sys
sys.path.append('../python')
import caffe
import yaml
import numpy as np
from multiprocessing import Queue
from .blob_fetchers import BlobFetcher
from lib.configuration import cfg, cfg_from_file, obj
import scipy.misc
import os.path as osp
import random
import logging
from cStringIO import StringIO
import pandas as pd
import scipy.io

class PrefetchDataLayer(caffe.Layer):
    """
    General Data Layer for batch of inputs.
    """
    def setup(self, bottom, top):
        """Setup layer:
        Initialize prefetching layer
        """
        param_dict = yaml.load(self.param_str)
        # setup_data will give class instance a BlobFetcher member
        self._blob_queue = Queue(10)
        self._prefetch_process = BlobFetcher(self._blob_queue, **param_dict)
        self._prefetch_process.start()

        def cleanup():
            '''
            This is termination function
            '''
            self._prefetch_process.terminate()
            self._prefetch_process.join()

        import atexit
        atexit.register(cleanup)
        # reshape data
        blobs = self.get_next_minbatch()
        for i in range(len(blobs)):
            top[i].reshape(*(blobs[i].shape))

    def get_next_minbatch(self):
        return self._blob_queue.get()

    def forward(self, bottom, top):
        blobs = self.get_next_minbatch()
        for i in range(len(blobs)):
            top[i].reshape(*(blobs[i].shape))
            # copy data into net's input blobs
            top[i].data[...] = blobs[i].astype(
                np.float32, copy=False)

    def backward(self, bottom, top, propagate_down):
        pass

    def reshape(self, bottom, top):
        """Reshape is taking place in forward"""
        pass


def substract_mean(img, mean):
    if len(mean) == 1:
        return (img - mean[0])
    elif len(mean) == 3:
        return (img - np.array(mean)[:, np.newaxis, np.newaxis])


class BaseImageVariationLayer(caffe.Layer):
    """Add variations to image

    Used for training denoising neural networks
    Top: [0]: produced image data of a patch size
         [1]: original image data of a patch size
    """
    def setup(self, bottom, top):
        layer_params = yaml.load(self.param_str)
        if 'cfg' not in layer_params.keys():
            logging.info("A configuration file must be provided")
            raise
        cfg_from_file(layer_params.get('cfg'))
        self._cfg = obj(cfg)
        self._batch_size = self._cfg.SAMPLING.BATCH_SIZE
        self._block_size = self._cfg.SAMPLING.BLOCK_SIZE
        self._mean = self._cfg.IMG_MEAN
        self._mirror = self._cfg.IMG_MIRROR
        self._force_color = self._cfg.FORCE_COLOR
        self._cls_label_provided = self._cfg.CLS_LABEL
        self.load_image_list()
        self._curr_idx = 0

        blob = self.get_next_minibatch()
        for i in range(len(blob)):
            top[i].reshape(*(blob[i].shape))

    def reshape(self, bottom, top):
        blob = self.get_next_minibatch()
        for i in range(len(blob)):
            top[i].reshape(*(blob[i].shape))
        self._blob = blob

    def load_image_list(self):
        """Load image list from cfg.IMAGE_LIST and cfg.IMG_ROOT"""
        if self._cfg.IMAGE_LIST == '' or not osp.exists(self._cfg.IMAGE_LIST):
            logging.info('A image list must be assigned')
            raise
        logging.info('Loading Image List into memory...')
        img_list_df = pd.read_csv(self._cfg.IMAGE_LIST,
                                  header=None, index_col=None,
                                  sep=' ')
        img_list_ = img_list_df[0].values
        # if len(img_list_df.columns) == 2:
        #     self._cls_label_provided = True
        #     img_labels_ = img_list_df[1].values.astype(np.int)
        #     self._cls_labels = img_labels_

        if self._cfg.IMG_ROOT != '' and osp.exists(self._cfg.IMG_ROOT):
            # img_list_ = [
            #     osp.join(self._cfg.IMG_ROOT, x.strip()) for x in img_list_]
            img_list_ = [
                self._cfg.IMG_ROOT + x.strip() for x in img_list_]
        # strip imglist
        img_list_ = [x.strip() for x in img_list_]

        # verify if file exists
        if self._cfg.DATA_TYPE == 'JPEG':
            if self._cfg.DEC_IMG:
                # read image into memory
                if self._force_color:
                    self._img_list = [
                        scipy.misc.imread(x, mode='RGB') for x in img_list_]
                else:
                    self._img_list = [
                        scipy.misc.imread(x) for x in img_list_]
            else:
                # only read compressed image as binary string
                self._img_list = []
                for x in img_list_:
                    with open(x, 'rb') as f:
                        self._img_list.append(f.read())
        elif self._cfg.DATA_TYPE == 'MAT':
            # self._img_list = []
            # for x in img_list_:
            #     d = scipy.io.loadmat(x)
            #     if d['image'].ndim == 2:
            #         self._img_list.append(np.tile(d['image'][:, :, None], (1, 1, 3)))
            #     else:
            #         self._img_list.append(d['image'])

            # Mat files would be loaded online
            self._img_list = [None] * len(img_list_)
        else:
            sys.exit('DATA_TYPE is not supported! Currently only support JPEG and MAT...')
        self._img_path_list = img_list_

        if len(img_list_df.columns) == 2:
            self._cls_label_provided = True
            if self._cfg.TASK_TYPE == 'CLASSIFICATION':
                img_labels_ = img_list_df[1].values.astype(np.int)
                self._cls_labels = img_labels_
            elif self._cfg.TASK_TYPE == 'SEGMENTATION':
                img_label_list_ = img_list_df[1].values
                self._cls_labels = [None] * len(img_label_list_)
                # if self._cfg.IMG_ROOT != '' and osp.exists(self._cfg.IMG_ROOT):
                #     # img_label_list_ = [
                #     #     osp.join(self._cfg.IMG_ROOT, x.strip()) for x in img_label_list_]
                #     img_label_list_ = [
                #         self._cfg.IMG_ROOT + x.strip() for x in img_label_list_]
                # # strip img_label_list
                # img_label_list_ = [x.strip() for x in img_label_list_]
                # self._cls_labels = [
                #     scipy.misc.imread(x, mode='L') for x in img_label_list_]
            else:
                sys.exit('TASK_TYPE is not supported! Currently only support CLASSIFICATION and SEGMENTATION...')

        logging.info("Totally %d images founded" % len(self._img_list))

    def apply_variation(self, img):
        """Abstract member function to implement for each class
        to add random variances to images

        Input image has already been normalized into [0, 1.0],
        which is performed in sample_blocks function
        """
        pass

    def sample_blocks(self, img):
        """Sample [count] blocks of variant / original blocks
        The variation is applied to each block individually
        """
        if self._cfg.SCALE <= 0:
            logging.error("Scalar must be positive")
            raise
        img = img.astype(np.float32) / self._cfg.SCALE
        self._blocks_per_img = int(self._batch_size / self._img_per_batch)
        img_blocks = []
        variantion_img_blocks = []

        # Check if we need to pad img to fit for crop_size
        new_height = max(self._block_size[0], img.shape[0])
        new_width = max(self._block_size[1], img.shape[1])

        new_img = np.ones((new_height, new_width, img.shape[2])) * np.array(self._mean)
        new_img[:img.shape[0], :img.shape[1], :] = img

        for i in range(self._blocks_per_img):
            x = random.randint(0, new_img.shape[0] - self._block_size[0])
            y = random.randint(0, new_img.shape[1] - self._block_size[1])
            block = new_img[x: x + self._block_size[0],
                            y: y + self._block_size[1],
                            :].transpose(2, 0, 1)
            # print (block.shape)
            variation_block = self.apply_variation(
                block.transpose(1, 2, 0)).transpose(2, 0, 1)
            crop = int(self._cfg.SAMPLING.CONV_CROP)
            if crop > 0:
                block = block[:, crop:-crop, crop:-crop]
                img_blocks.append(block)
            else:
                img_blocks.append(block)
            variantion_img_blocks.append(variation_block)
        return img_blocks, variantion_img_blocks

    def get_next_minibatch(self):
        """Generate next mini-batch

        The return value is array of numpy array:
        [variation_image, original_img]
        Reshape funcion will be called based on resutls of this function

        Sampling:
        For each mini-batch
        1. Sampling a certain number of images
        2. Sample certain amount of blocks from each image
        3. Apply image variations to each block

        Classification Label:
        If cfg.CLS_LABEL is Ture, output the original classification label
        If cfg.ORIGINAL_IMG_LABEL is True, output the original image patch
        """
        self._img_per_batch = self._cfg.SAMPLING.IMG_PER_BATCH

        if self._cfg.SHUFFLE:
            # randomly sampling image
            img_list_idx = random.sample(range(len(self._img_list)),
                                         self._img_per_batch)
        else:
            if self._curr_idx + self._img_per_batch > len(self._img_path_list):
                _next_idx = self._curr_idx + self._img_per_batch - len(self._img_path_list)
                img_list_idx = range(self._curr_idx, len(self._img_path_list)) + range(_next_idx)
                self._curr_idx = _next_idx
            else:
                img_list_idx = range(self._curr_idx, self._curr_idx + self._img_per_batch)
                self._curr_idx += self._img_per_batch
        img_blocks = []
        variation_blocks = []
        cls_labels = []
        print ('img_list_idx: ', img_list_idx)
        for idx in img_list_idx:
            if self._cfg.DATA_TYPE == 'MAT':
                d = scipy.io.loadmat(self._img_path_list[idx])
                if d['image'].ndim == 2:
                    img = np.tile(d['image'][:, :, None], (1, 1, 3))
                else:
                    img = d['image']
            else:
                img = self._img_list[idx]
            if type(img) is str:
                # turn string into numpy array
                if self._force_color:
                    img = scipy.misc.imread(StringIO(img), mode='RGB')
                else:
                    img = scipy.misc.imread(StringIO(img))
            if img.ndim == 3:
                # Flip color channel for input image
                img = img[:, :, ::-1]
            # if img.shape[0] <= self._block_size[0] or \
            #    img.shape[1] <= self._block_size[1]:
            #     continue
            if self._mirror and random.random() > 0.5:
                img = np.fliplr(img)
            r = self.sample_blocks(img)
            img_blocks.append(r[0])
            variation_blocks.append(r[1])
            if self._cls_label_provided:
                cls_label = self._cls_labels[idx]
                for i in range(self._blocks_per_img):
                    cls_labels.append(cls_label)
        img_blocks = sum(img_blocks, [])
        variation_blocks = sum(variation_blocks, [])
        res = (substract_mean(np.array(variation_blocks), self._mean),)
        if self._cfg.CLS_LABEL:
            res += (np.array(cls_labels).reshape(len(cls_labels), 1),)
        if self._cfg.ORIGINAL_IMG_LABEL:
            res += (substract_mean(np.array(img_blocks), self._mean),)
        # return value:
        if len(res) == 1:
            # return res[0]
            return res
        else:
            return res

    def backward(self, bottom, top):
        pass

    def forward(self, bottom, top):
        # blob = self.get_next_minibatch()
        blob = self._blob
        for i in range(len(blob)):
            top[i].reshape(*blob[i].shape)
            top[i].data[...] = blob[i].astype(np.float32, copy=False)
