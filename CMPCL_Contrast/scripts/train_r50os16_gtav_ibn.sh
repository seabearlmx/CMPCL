#!/usr/bin/env bash
    # Example on gtav
     python -m torch.distributed.launch --nproc_per_node=4 train.py \
        --dataset gtav \
        --covstat_val_dataset gtav \
        --val_dataset cityscapes bdd100k mapillary synthia \
        --arch network.deepv3.DeepR50V3PlusD \
        --city_mode 'train' \
        --lr_schedule poly \
        --lr 0.01 \
        --poly_exp 0.9 \
        --max_cu_epoch 10000 \
        --class_uniform_pct 0.5 \
        --class_uniform_tile 1024 \
        --crop_size 768 \
        --scale_min 0.5 \
        --scale_max 2.0 \
        --rrotate 0 \
        --max_iter 40000 \
        --bs_mult 8 \
        --gblur \
        --color_aug 0.5 \
        --wt_reg_weight 0.0 \
        --relax_denom 0.0 \
        --cov_stat_epoch 0 \
        --wt_layer 0 0 4 4 4 0 0 \
        --date 0201 \
        --exp r50os16_gtav_ibn \
        --ckpt ./logs/ \
        --tb_path ./logs/
