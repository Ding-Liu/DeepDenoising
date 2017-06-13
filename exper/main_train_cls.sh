#!/usr/bin/env bash
python train_net_joint.py --solver=config/solver_cls.prototxt --weights=model/denoise-s30.caffemodel --weights_2=model/cls_init.caffemodel --GPU=0
