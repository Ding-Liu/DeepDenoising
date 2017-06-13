"""
Configuration for noisy image layer and denoising network
Including noisy pattern / types for network training,
training settings, and testing settings.

Some functions are from Girshick's Fast-RCNN github repo
Replace the easydict to yaml and dict
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import yaml


__authors__ = ['Xianming Liu (xmliu@fb.com)']

__C = {}
cfg = __C
#
# Layer Parameter
#
# Image List file path, must be provided through input file
__C['IMAGE_LIST'] = ''
__C['IMG_ROOT'] = ''

# BATCH_SIZE
__C['BATCH_SIZE'] = 512

# shuffle data or not
__C['SHUFFLE'] = False

# NOISE TYPE: default is all (use gaussian, s&p, and poisson noise)
# And noise parameters
__C['NOISE_TYPE'] = ['gaussian', 's&p', 'poisson']
__C['GAUSSIAN_MAG'] = 1.0
__C['GAUSSIAN_SIGMA'] = 10.0
__C['S_VS_P'] = 0.5
__C['SP_AMOUNT'] = 0.004
__C['IMG_MEAN'] = [0.0]
__C['IMG_MIRROR'] = False
__C['FORCE_COLOR'] = True
__C['SCALE'] = 1.0

# DATA TYPE: default input data type is JPEG, MAT data can be handled as well
__C['DATA_TYPE'] = 'JPEG'

# TASK_TYPE: default task is image classification, image segmentation can be handled as well
__C['TASK_TYPE'] = 'CLASSIFICATION'

# HAZING IMAGE PARAMETERS
# ATMO_LIGHT: global atmospheric light (A)
# MEDIUM_TRAN: medium transmission (t(x))
# Hazy image model: I(x) = J(x) * t(x) + A(1 - t(x))
__C['MAX_ATMO_LIGHT'] = 100
__C['MAX_MEDIUM_TRAN'] = 0.6

# Composed controls if both noisy and hazy being added to images
__C['COMPOSED'] = False

# Type of output
__C['FLATTEN'] = True

# Control if the classification label will be outputed
__C['CLS_LABEL'] = False
__C['ORIGINAL_IMG_LABEL'] = True
__C['DEC_IMG'] = False

# SAMPLING related configuratons
# LABEL_DOWNSAMPLING: used for sampling labels with downsampled ratio
__C['SAMPLING'] = {}
__C['SAMPLING']['IMG_PER_BATCH'] = 16
__C['SAMPLING']['BLOCK_SIZE'] = [13, 13]
__C['SAMPLING']['BATCH_SIZE'] = 512
__C['SAMPLING']['LABEL_DOWNSAMPLING'] = 1
# CONV_CROP: is the margin cropped by applying convolution
# if conv_crop > 0, then the label will be a smaller patch
# with size block_size - 2 x conv_crop
__C['SAMPLING']['CONV_CROP'] = 0

# Test settings, used for denoising large image
__C['TEST'] = {}
__C['TEST']['NETWORK_NAME'] = ''
__C['TEST']['NETWORK_FN'] = ''
__C['TEST']['MODEL_FN'] = ''
__C['TEST']['INPUT_BLOB_NAME'] = 'data'
__C['TEST']['OUTPUT_BLOB_NAME'] = 'output'
__C['TEST']['STRIDE'] = 1
# if block size is -1, then use the corresponding image dimension
__C['TEST']['BLOCK_SIZE'] = [17, 17]
__C['TEST']['DEVICE'] = 'GPU'
__C['TEST']['DEVICE_ID'] = 0
__C['TEST']['MAX_BATCH_SIZE'] = 512


def _merge_a_into_b(a, b):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    if type(a) is not dict:
        return

    for k, v in a.iteritems():
        # a must specify keys that are in b
        if k not in b:
            raise KeyError('{} is not a valid config key'.format(k))

        # the types must match, too
        old_type = type(b[k])
        if old_type is not type(v):
            if isinstance(b[k], np.ndarray):
                v = np.array(v, dtype=b[k].dtype)
            else:
                raise ValueError(('Type mismatch ({} vs. {}) '
                                  'for config key: {}').format(
                                      type(b[k]), type(v), k))

        # recursively merge dicts
        if type(v) is dict:
            _merge_a_into_b(a[k], b[k])
        else:
            b[k] = v


def cfg_from_file(filename):
    """Load a config file and merge it into the default options."""
    with open(filename, 'r') as f:
        yaml_cfg = yaml.load(f)
    _merge_a_into_b(yaml_cfg, __C)


class obj(object):
    def __init__(self, d):
        for a, b in d.items():
            if isinstance(b, (list, tuple)):
                setattr(self, a,
                        [obj(x) if isinstance(x, dict) else x for x in b])
            else:
                setattr(self, a, obj(b) if isinstance(b, dict) else b)
