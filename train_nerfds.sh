#!/bin/bash

python train.py \
  -s /home/old/gaussian4d/workspace/4DGaussians/data/nerf-ds/press \
  --port 6017 \
  --expname "nerf-ds/press" \
  --configs arguments/dsnerf/press.py \
  --rlambda 0.001
