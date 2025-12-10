#!/bin/bash

# First, computing dense point clouds by COLMAP
bash colmap.sh data/hypernerf/virg/broom2 hypernerf

# Second, downsample the point clouds generated in the first step.
python scripts/downsample_point.py \
  data/hypernerf/virg/broom2/colmap/dense/workspace/fused.ply \
  data/hypernerf/virg/broom2/points3D_downsample2.ply

# Finally, train.
python train.py \
  -s data/hypernerf/virg/broom2/ \
  --port 6017 \
  --expname "hypernerf/broom2" \
  --configs arguments/hypernerf/broom2.py \
  --rlambda 0.001
