#!/bin/bash

# First, extract the frames of each video.
python scripts/preprocess_dynerf.py --datadir data/dynerf/cut_roasted_beef

# Second, generate point clouds from input data.
bash colmap.sh data/dynerf/cut_roasted_beef llff

# Third, downsample the point clouds generated in the second step.
python scripts/downsample_point.py \
  data/dynerf/cut_roasted_beef/colmap/dense/workspace/fused.ply \
  data/dynerf/cut_roasted_beef/points3D_downsample2.ply

# Finally, train.
python train.py \
  -s data/dynerf/cut_roasted_beef \
  --port 6017 \
  --expname "dynerf/cut_roasted_beef" \
  --configs arguments/dynerf/cut_roasted_beef.py \
  --rlambda 0.001
