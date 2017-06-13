#!/usr/bin/env sh
# This script converts the cifar data into leveldb format.
export LD_LIBRARY_PATH="/usr/local/cuda-7.5-please-use-cuda-8.0/lib64:/usr/local/lib:/home/dingliu2/Documents/cudnn_v3:$LD_LIBRARY_PATH"


EXAMPLE=examples/cifar10
DATA=data/cifar10
DBTYPE=lmdb

echo "Creating $DBTYPE..."

rm -rf $EXAMPLE/cifar10_train_$DBTYPE $EXAMPLE/cifar10_test_$DBTYPE

./build/examples/cifar10/convert_cifar_data.bin $DATA $EXAMPLE $DBTYPE

echo "Computing image mean..."

./build/tools/compute_image_mean -backend=$DBTYPE \
  $EXAMPLE/cifar10_train_$DBTYPE $EXAMPLE/mean.binaryproto

echo "Done."
