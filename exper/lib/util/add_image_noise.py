"""Add noise to image

Type of noise:
- Gaussian
- Salt & Pepper (s&p)
- Poisson noise
- Speckle (mixed noise)
"""
import numpy as np
import yaml


def noise(noise_typ, image, **kwargs):
    if kwargs:
        # turn kwargs into yaml
        param = yaml.load(str(kwargs))
    if noise_typ == "gaussian":
        row, col, ch = image.shape
        mean = param.get('mean', 0)
        var = param.get('var', 0.1)
        if var == 0.0:
            # if sigma is 0, return noise free image
            return image
        mag = param.get('magnitude', 1)
        sigma = var**0.5
        gauss = np.random.normal(mean, sigma, (row, col, ch))
        gauss = gauss.reshape(row, col, ch)
        noisy = image + mag * gauss
        return noisy
    elif noise_typ == "s&p":
        row, col, ch = image.shape
        s_vs_p = param.get('s_vs_p', 0.5)
        amount = param.get('amount', 0.004)
        out = image
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
                  for i in image.shape]
        out[coords] = 1

        # Pepper mode
        num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
                  for i in image.shape]
        out[coords] = 0
        return out
    elif noise_typ == "poisson":
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(image * vals) / float(vals)
        return noisy
    elif noise_typ =="speckle":
        row, col, ch = image.shape
        gauss = np.random.randn(row, col, ch)
        gauss = gauss.reshape(row, col, ch)
        noisy = image + image * gauss
        return noisy
