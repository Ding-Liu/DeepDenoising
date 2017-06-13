#!/usr/bin/env bash
python train_net_joint.py --solver=config/solver_seg.prototxt --weights=model/denoise-s30.caffemodel --weights_2=model/seg_init.caffemodel --GPU=0
