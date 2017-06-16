#!/usr/bin/env bash
export PYTHONPATH='../python/':$PYTHONPATH
python tools/deepdenoising.py --cfg=yml/denoise_deploy.yml --image=data/0010x4.png  --sigma=30 --evaluate
