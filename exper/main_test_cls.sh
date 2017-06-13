#!/usr/bin/env bash
export PYTHONPATH='../python/':$PYTHONPATH
../build/tools/caffe test -model config/test_cls.prototxt -weights model/cls_joint.caffemodel -gpu 0 -iterations 5000
