#!/bin/bash

python train.py \
  -s /home/old/gaussian4d/workspace/4DGaussians/data/dsnerf/press \
  --port 6017 \
  --expname "dsnerf/press" \
  --configs arguments/dsnerf/press.py \
  --rlambda 0.001
