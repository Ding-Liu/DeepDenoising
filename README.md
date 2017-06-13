# DeepDenoising

## Introduction
This repository provides codes that we use to study the mutual influence between image denoising and high-level vision tasks.

(1) We present an image denoising network which achieves state-of-the-art image denoising performance. 

(2) We propose a deep network solution that cascades two modules for image denoising and various high-level tasks, respectively, and demonstrate that the proposed architecture not only yields superior image denoising results preserving fine details, but also overcomes the performance degradation of different high-level vision tasks, such as image classification and semantic segmentation, due to image noise.

This code repository is built on top of [DeepLab v2](https://bitbucket.org/aquariusjay/deeplab-public-ver2).

### Download models
- `cd exper/model/`
- Run `get_models.sh` to download models used in our work.

### Training
- `cd exper`
- Run `main_train_denoise.sh` to train the denoising network seperately.
- Run `main_train_cls.sh` to jointly train the cascade of the denoising network and the network for image classification.
- Run `main_train_seg.sh` to jointly train the cascade of the denoising network and the entwork for semantic segmentation.

### Testing
- `cd exper`
- Run `main_test_cls.sh` to test the resulting model for image classification.
- Run `main_test_seg.sh` to test the resulting model for semantic segmentation.
