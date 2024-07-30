## Train
```
<path_to_robustnet>$ CUDA_VISIBLE_DEVICES=0,1,2,3 ./scripts/train_r50os16_gtav_isw.sh # Train: GTAV, Test: BDD100K, Cityscapes, Synthia, Mapillary / ResNet50, ISW
<path_to_robustnet>$ CUDA_VISIBLE_DEVICES=0,1,2,3 ./scripts/train_r50os16_gtav_ibn.sh # Train: GTAV, Test: BDD100K, Cityscapes, Synthia, Mapillary / ResNet50, IBN-Net
<path_to_robustnet>$ CUDA_VISIBLE_DEVICES=0,1,2,3 ./scripts/train_r50os16_gtav_base.sh # Train: GTAV, Test: BDD100K, Cityscapes, Synthia, Mapillary / ResNet50, Baseline (DeepLabV3+)
```

## Evaluation
```
<path_to_robustnet>$ CUDA_VISIBLE_DEVICES=0,1,2,3 ./scripts/valid_r50os16_gtav_isw.sh # Train: GTAV, Test: BDD100K, Cityscapes, Synthia, Mapillary / ResNet50, ISW
<path_to_robustnet>$ CUDA_VISIBLE_DEVICES=0,1,2,3 ./scripts/valid_r50os16_gtav_ibn.sh # Train: GTAV, Test: BDD100K, Cityscapes, Synthia, Mapillary / ResNet50, IBN-Net
<path_to_robustnet>$ CUDA_VISIBLE_DEVICES=0,1,2,3 ./scripts/valid_r50os16_gtav_base.sh # Train: GTAV, Test: BDD100K, Cityscapes, Synthia, Mapillary / ResNet50, Baseline (DeepLabV3+)
```

## Acknowledgments
Our pytorch implementation is heavily derived from [RobustNet](https://github.com/shachoi/RobustNet).
