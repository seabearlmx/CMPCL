#!/usr/bin/env bash
echo "Running inference on" ${1}

     python -m torch.distributed.launch --nproc_per_node=4 valid.py \
        --val_dataset synthia bdd100k cityscapes mapillary \
        --arch network.deepv3.DeepShuffleNetV3PlusD \
        --wt_layer 0 0 2 2 2 0 0 \
        --date 0101 \
        --exp shuffle_gtav_isw \
        --snapshot ${1}
