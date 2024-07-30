syn_type = "CityscapesDataset"
syn_root = "/extra_disk/Rein/SYNTHIA/data/"
syn_crop_size = (512, 512)
syn_train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations"),
    dict(type="Resize", scale=(1280, 760)),
    dict(type="RandomCrop", crop_size=syn_crop_size, cat_max_ratio=0.75),
    dict(type="RandomFlip", prob=0.5),
    dict(type="PhotoMetricDistortion"),
    dict(type="PackSegInputs"),
]
syn_test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="Resize", scale=(1280, 760), keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type="LoadAnnotations"),
    dict(type="PackSegInputs"),
]
train_syn = dict(
    type=syn_type,
    data_root=syn_root,
    data_prefix=dict(
        img_path="RGB",
        seg_map_path="GT/LABELS",
    ),
    img_suffix=".png",
    seg_map_suffix="_labelTrainIds.png",
    pipeline=syn_train_pipeline,
)
val_syn = dict(
    type=syn_type,
    data_root=syn_root,
    data_prefix=dict(
        img_path="RGB",
        seg_map_path="GT/LABELS",
    ),
    img_suffix=".png",
    seg_map_suffix="_labelTrainIds.png",
    pipeline=syn_test_pipeline,
)