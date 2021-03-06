#!/usr/bin/env bash

# Test GenRe

out_dir="./inpaint_refine_output"
fullmodel=./downloads/models/full_model.pt
rgb_pattern='./emptydir/*_rgb.*'
mask_pattern='./emptydir/*_silhouette.*'

if [ $# -lt 1 ]; then
    echo "Usage: $0 gpu[ ...]"
    exit 1
fi
gpu="$1"
shift # shift the remaining arguments

set -e

source activate shaperecon

python 'inpaint_refine.py' \
    --net genre_full_model \
    --net_file "$fullmodel" \
    --input_rgb "$rgb_pattern" \
    --input_mask "$mask_pattern" \
    --output_dir "$out_dir" \
    --suffix '{net}' \
    --overwrite \
    --workers 0 \
    --batch_size 1 \
    --vis_workers 4 \
    --gpu "$gpu" \
    $*

conda deactivate
