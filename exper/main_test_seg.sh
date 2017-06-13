#!/usr/bin/env bash
export PYTHONPATH='../python/':$PYTHONPATH
../build/tools/caffe test --model=config/test_seg.prototxt --weights=model/seg_joint.caffemodel --gpu=0 --iterations=1449
