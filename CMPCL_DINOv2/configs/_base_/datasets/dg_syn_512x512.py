_base_ = [
    "./syn_512x512.py",
    "./bdd100k_512x512.py",
    "./cityscapes_512x512.py",
    "./mapillary_512x512.py",
]
train_dataloader = dict(
    batch_size=8,
    num_workers=4,
    persistent_workers=False,
    pin_memory=False,
    sampler=dict(type="InfiniteSampler", shuffle=True),
    dataset={{_base_.train_syn}},
)
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=False,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type="ConcatDataset",
        datasets=[
            {{_base_.val_cityscapes}},
            {{_base_.val_bdd}},
            {{_base_.val_mapillary}},
        ],
    ),
)
test_dataloader = val_dataloader
val_evaluator = dict(
    type="DGIoUMetric", iou_metrics=["mIoU"], dataset_keys=["citys", "map", "bdd"]
)
test_evaluator=val_evaluator
