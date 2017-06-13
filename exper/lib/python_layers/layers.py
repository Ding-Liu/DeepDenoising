"""Python Layer: NoisyImageLayer

Add random noise to image.
Being used in training denoising network
"""

from .base_py_layer import BaseImageVariationLayer
from lib.util.add_image_noise import noise
import numpy as np
import random


class NoisyImageLayer(BaseImageVariationLayer):
    """Add random noise to image

    Used for training denoising neural networks
    Top: [0]: noisy image data of a patch size
         [1]: original image data of a patch size
    """
    def apply_variation(self, img):
        # choose noise type:
        noise_type = random.choice(self._cfg.NOISE_TYPE)
        kwargs = {}
        gaussian_var = (float(self._cfg.GAUSSIAN_SIGMA) / self._cfg.SCALE) ** 2
        kwargs['var'] = gaussian_var
        kwargs['magnitude'] = self._cfg.GAUSSIAN_MAG
        kwargs['s_vs_p'] = self._cfg.S_VS_P
        kwargs['amount'] = self._cfg.SP_AMOUNT
        noisy_image = noise(noise_type, img, **kwargs)
        return noisy_image


class HazyImageLayer(BaseImageVariationLayer):
    """Add haze to images / image patchs
    which is used to train dehazing network
    """
    def apply_variation(self, img):
        max_atmo_light = float(self._cfg.MAX_ATMO_LIGHT) / 255.0
        max_medium_tran = float(self._cfg.MAX_MEDIUM_TRAN)

        # start generating haze model
        """
        _A = np.random.rand() * max_atmo_light
        """
        _alpha = np.random.rand() * max_medium_tran
        _A = max_atmo_light
        hazy_img = img * (1 - _alpha) + _A * _alpha

        if self._cfg.COMPOSED:
            # also add noise into the hazy images
            noise_type = random.choice(self._cfg.NOISE_TYPE)
            kwargs = {}
            gaussian_var = (float(self._cfg.GAUSSIAN_SIGMA) / 255.0) ** 2
            kwargs['var'] = gaussian_var
            kwargs['magnitude'] = self._cfg.GAUSSIAN_MAG
            kwargs['s_vs_p'] = self._cfg.S_VS_P
            kwargs['amount'] = self._cfg.SP_AMOUNT
            hazy_img = noise(noise_type, hazy_img, **kwargs)
        return hazy_img
