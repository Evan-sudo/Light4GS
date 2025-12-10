#!/bin/bash

python train.py \
  -s data/dnerf/bouncingballs \
  --port 6017 \
  --expname "dnerf/bouncingballs" \
  --configs arguments/dnerf/bouncingballs.py \
  --rlambda 0.001
