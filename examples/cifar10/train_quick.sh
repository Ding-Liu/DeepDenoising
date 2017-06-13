#!/usr/bin/env sh
export LD_LIBRARY_PATH="/usr/local/cuda-7.5-please-use-cuda-8.0/lib64:/usr/local/lib:/home/dingliu2/Documents/cudnn_v3:$LD_LIBRARY_PATH"


TOOLS=./build/tools

$TOOLS/caffe train \
  --solver=examples/cifar10/cifar10_quick_solver.prototxt

# reduce learning rate by factor of 10 after 8 epochs
#$TOOLS/caffe train \
#  --solver=examples/cifar10/cifar10_quick_solver_lr1.prototxt \
#  --snapshot=examples/cifar10/cifar10_quick_iter_4000.solverstate.h5
