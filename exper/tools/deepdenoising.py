"""Deep Denoising Main Script
Image Denoising using Deep Convolutional Neural Network
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
fwd = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(fwd, os.pardir, 'lib'))
import scipy.misc
from denoise_image import ImageDenoiser
from configuration import cfg
from util import (load_image, computePSNR, gray2rgb)
import argparse
import time
import numpy as np
from PIL import Image


def main():
    parser = argparse.ArgumentParser("Denosing image using neural network")
    parser.add_argument(
        '--cfg', help='Configuration file',
        required=True)
    parser.add_argument(
        '--image', help='Image file to denoise',
        required=True)
    parser.add_argument(
        '--output_path', help='Output file path',
        default='./eval/')
    parser.add_argument(
        '-e', '--evaluate', action='store_true',
        help="Running evaluation of PSNR", default=False)
    parser.add_argument(
        '--sigma', default=25,
        help="Standard deviation for generating noise during evaluation")
    parser.add_argument(
        '--noisy_image',
        help="Noisy image for evaluation")
    arg = parser.parse_args()

    if not os.path.exists(arg.output_path):
        os.mkdir(arg.output_path)

    denoiser = ImageDenoiser(arg.cfg)

    img = load_image(arg.image)
    file_name = os.path.basename(arg.image)
    file_basename, file_extension = os.path.splitext(file_name)

    if arg.evaluate:
        if arg.noisy_image:
            noisy_img = scipy.misc.imread(arg.noisy_image).astype(np.float)
            # noisy_img /= 255
        else:
            # add noise and run evaluation of psnr
            mean = 0
            sigma = float(arg.sigma)
            noise = np.random.normal(mean, sigma, img.shape)
            noisy_img = img + noise
            # scipy.misc.imsave(
            #     os.path.join(arg.output_path, file_basename + '_noisy.png'),
            #     np.clip(noisy_img, 0, 255))
    else:
        noisy_img = img

    # start denoising
    start = time.time()
    denoised_image = denoiser.denoise(noisy_img)
    print("Time cost: {} second".format(time.time() - start))

    # Test: using Image to save file
    denoised_img = Image.fromarray(denoised_image.astype(np.uint8))
    denoised_img.save(
        arg.output_path + file_basename + '_denoised.png',
        format='PNG'
    )
    # denoised_img.save(
    #     arg.output_path + file_basename + '_denoised.jpg',
    #     format='JPEG', quality=100
    # )
    """
    scipy.misc.imsave(
        arg.output_path + file_basename + '_denoised.png',
        np.clip(denoised_image, 0, 1))
    """
    print('Denoising Finished!')

    if arg.evaluate:
        if img.ndim == 2:
            img = gray2rgb(img)
            noisy_img = gray2rgb(noisy_img)
            color_channel = False
        else:
            color_channel = True
        rmse, psnr = computePSNR(
            img, denoised_image, cfg, ColorChannel=color_channel)
        rmse_n, psnr_n = computePSNR(
            img, noisy_img, cfg, ColorChannel=color_channel)
        print('%s: RMSE = %.4f / %.4f, PSNR = %.4f / %.4f' %
              (file_basename, rmse, rmse_n, psnr, psnr_n))

if __name__ == "__main__":
    main()
