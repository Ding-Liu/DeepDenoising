from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import sys
import numpy as np
import os.path as osp
import random
from cStringIO import StringIO
from multiprocessing import Process
from PIL import Image
from skimage import io
from lib.configuration import cfg, cfg_from_file, obj
import glob
import logging


# start definition of DataProvider
class SynthesizedGrayDataProvider(object):
    def __init__(self, **kwargs):
        print(kwargs)
        self._batch_size = int(kwargs.get('batchsize', 64))
        self._img_list_fn = kwargs.get('source')
        self._root = kwargs.get('root', None)
        with open(self._img_list_fn, 'rb') as fp:
            lines = fp.readlines()
            print("Reading Image List Completed")
        self._img_list = map(lambda x: x.strip().split(' '), lines)
        self._img_list = map(
            lambda x: (
                osp.join(self._root, x[0]) if self._root is not None
                else x[0],
                int(x[1])), self._img_list)
        # filter out non-existing imags
        # self._img_list = [x for x in self._img_list if osp.exists(x[0])]
        self._sample_count = len(self._img_list)
        print("Totally {} images found".format(self._sample_count))
        self._cur = 0
        self._shuffle = kwargs.get('shuffle', False)
        if self._shuffle:
            random.shuffle(self._img_list)
        self._img_mean = kwargs.get('mean', None)

    def get_next_batch(self):
        """Get next batch of self._batch_size
        """
        data = []
        label = []
        for i in range(self._batch_size):
            try:
                img_fn, img_label = self._img_list[self._cur]
                self._cur = (self._cur + 1) % self._sample_count
                img = Image.open(img_fn)
                label.append(int(img_label))
                # change to gray and change back to RGB
                img = synthesized_gray(img)
                data.append(get_datum(img, self._img_mean))
            except:
                print("Image file does not exist: {}".format(img_fn))
                continue
        data = np.array(data)
        label = np.array(label).reshape(self._batch_size, 1, 1, 1)
        return [data, label]


class BaseSemanticSegmentationDataProvider(object):
    """Class used in training semantic segmenation using caffe

    The input will indicate two folder:
    1. directory for original image
    2. directory for labeled maps

    Data association: same basename
    Reading parameter from cfg file
    Parameters including:
      - cfg.SAMPLING.IMG_PER_BATCH
      - cfg.SAMPLING.BLOCK_SIZE
    Batchsize is indicated in kwargs
    """
    def __init__(self, **kwargs):
        print(kwargs)
        self._batch_size = int(kwargs.get('batchsize', 64))
        self._img_dir = kwargs.get('images')
        self._label_dir = kwargs.get('labels')
        self._ext = kwargs.get('extension', 'jpg')
        self._label_ext = kwargs.get('label ext', self._ext)
        self._root = kwargs.get('root', None)
        if self._root is not None:
            self._img_dir = osp.join(self._root, self._img_dir)
            self._label_dir = osp.join(self._root, self._label_dir)
        # parse image list: information in self._img_list
        self._parse_image_list()
        self._sample_count = len(self._img_list)
        print("Totally {} images found".format(self._sample_count))
        self._cur = 0
        self._shuffle = kwargs.get('shuffle', False)
        if self._shuffle:
            random.shuffle(self._img_list)
        self._img_mean = kwargs.get('mean', None)
        cfg_from_file(kwargs.get('cfg'))
        self._cfg = obj(cfg)
        self._pre_read_data()

    def _parse_image_list(self):
        """Get image list from two directories: images and labels
        """
        fns = glob.glob(osp.join(self._img_dir, '*.{}'.format(self._ext)))
        fns = [osp.basename(x).rstrip('.{}'.format(self._ext)) for x in fns]
        self._img_list = filter(
            lambda x: osp.exists(
                osp.join(self._label_dir, "{}.{}".format(x, self._label_ext))
            ), fns)

    def _pre_read_data(self):
        self._data = []
        self._labels = []
        logging.info("Starting pre loading data from disk to memory")
        for x in self._img_list:
            _data_fn = osp.join(self._img_dir, "{}.{}".format(
                x, self._ext))
            _label_fn = osp.join(self._label_dir, "{}.{}".format(
                x, self._label_ext))
            with open(_data_fn, 'rb') as fp:
                self._data.append(fp.read())
            with open(_label_fn, 'rb') as fp:
                self._labels.append(fp.read())
        logging.info("File pre loading finished")

    def sampling_image(self, data, label):
        """
        This function needs to be implemted for each class
        based on the BaseSemanticSegmentationDataProvider class
        """
        # TODO: Implement sampling function:
        # output data blob and pixel-wise labels
        pass

    def get_next_batch(self):
        """Get next batch of self._batch_size
        """
        data = []
        label = []
        self._sample_per_image = self._batch_size / int(
            self._cfg.SAMPLING.IMG_PER_BATCH)
        for i in range(int(self._cfg.SAMPLING.IMG_PER_BATCH)):
            try:
                img_ = get_datum(self._data[self._cur], self._img_mean)
                label_ = get_datum(self._labels[self._cur], None)
                sampled_datum, sampled_label = self.sampling_image(img_, label_)
                label.append(sampled_label)
                data.append(sampled_datum)
                self._cur = (self._cur + 1) % self._sample_count
            except:
                logging.info("Errors in decoding and sampling image")
                continue
        data = np.vstack(data)
        label = np.vstack(label)
        return [data, label]


class BlobFetcher(Process):
    def __init__(self, queue, **kwargs):
        '''
        Initilizing a blob fetcher
        Note:
        data_provier: either for training or testing
        '''
        super(BlobFetcher, self).__init__()
        print("Staring Blob Fetecher...")
        self.name = "BlobFetcher"
        self._queue = queue
        self.setup_data(**kwargs)

    def setup_data(self, **kwargs):
        data_provder_type = kwargs.get('DataProvider')
        print("Data Provider Type: {}".format(data_provder_type))
        if data_provder_type == "SynthesizedGrayImage":
            self._dataprovider = SynthesizedGrayDataProvider(**kwargs)

    def get_next_batch(self):
        """Need to implement to get one batch"""
        return self._dataprovider.get_next_batch()

    def run(self):
        '''
        This function defines what each fetcher does
        '''
        print("BlobFetcher Started")
        # load the next batch and put it into the queue
        while True:
            blobs = self.get_next_batch()
            self._queue.put(blobs)


def get_datum(img, image_mean=None):
    try:
        # if input is a file name, then read image; otherwise decode_imgstr
        if type(img) is np.ndarray:
            img_data = img
        else:
            img_data = decode_imgstr(img)
        img_data = img_data.astype(np.float32, copy=False)
        img_data = img_data[:, :, ::-1]
        # change channel for caffe:
        img_data = img_data.transpose(2, 0, 1)  # to CxHxW
        # substract_mean
        if image_mean is not None:
            img_data = substract_mean(img_data, image_mean)
        return img_data
    except:
        print(sys.exc_info()[0], sys.exc_info()[1])
        return


def decode_imgstr(imgstr):
    img_data = io.imread(StringIO(imgstr))
    return img_data


def substract_mean(img, image_mean):
    """Substract image mean from data sample
    image_mean is a numpy array,
    either 1 * 3 or of the same size as input image
    """
    if image_mean.ndim == 1:
        image_mean = image_mean[:, np.newaxis, np.newaxis]
    img -= image_mean
    return img


def synthesized_gray(img):
    gray_img = img.convert('L')
    synthesized_gray_img = gray_img.convert('RGB')
    return np.array(synthesized_gray_img)
