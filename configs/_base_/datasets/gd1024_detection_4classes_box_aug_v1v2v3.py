dataset_type = 'Gd1024Dataset'
classes = ('1', '2', '3', '4')
data_root = 'data/gd_newAug5_Rot0_4classes/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
train_data1 = dict(
        classes=classes,
        type='Gd1024Dataset',
        ann_file='data/gd_newAug5_Rot0_4classes/box_aug_v1/train1/train1.json',
        img_prefix='data/gd_newAug5_Rot0_4classes/box_aug_v1/train1/images/',
        pipeline=train_pipeline)
val_data1 = dict(
        classes=classes,
        type='Gd1024Dataset',
        ann_file='data/gd_newAug5_Rot0_4classes/box_aug_v1/val1/val1.json',
        img_prefix='data/gd_newAug5_Rot0_4classes/box_aug_v1/val1/images/',
        pipeline=test_pipeline)
test_data1 = dict(
        classes=classes,
        type='Gd1024Dataset',
        ann_file='data/gd_newAug5_Rot0_4classes/box_aug_v1/val1/val1.json',
        img_prefix='data/gd_newAug5_Rot0_4classes/box_aug_v1/val1/images/',
        pipeline=test_pipeline)

train_data2 = dict(
        classes=classes,
        type='Gd1024Dataset',
        ann_file='data/gd_newAug5_Rot0_4classes/box_aug_v2/train1/train1.json',
        img_prefix='data/gd_newAug5_Rot0_4classes/box_aug_v2/train1/images/',
        pipeline=train_pipeline)
val_data2 = dict(
        classes=classes,
        type='Gd1024Dataset',
        ann_file='data/gd_newAug5_Rot0_4classes/box_aug_v2/val1/val1.json',
        img_prefix='data/gd_newAug5_Rot0_4classes/box_aug_v2/val1/images/',
        pipeline=test_pipeline)
test_data2 = dict(
        classes=classes,
        type='Gd1024Dataset',
        ann_file='data/gd_newAug5_Rot0_4classes/box_aug_v2/val1/val1.json',
        img_prefix='data/gd_newAug5_Rot0_4classes/box_aug_v2/val1/images/',
        pipeline=test_pipeline)

train_data3 = dict(
        classes=classes,
        type='Gd1024Dataset',
        ann_file='data/gd_newAug5_Rot0_4classes/box_aug_v3/train1/train1.json',
        img_prefix='data/gd_newAug5_Rot0_4classes/box_aug_v3/train1/images/',
        pipeline=train_pipeline)
val_data3 = dict(
        classes=classes,
        type='Gd1024Dataset',
        ann_file='data/gd_newAug5_Rot0_4classes/box_aug_v3/val1/val1.json',
        img_prefix='data/gd_newAug5_Rot0_4classes/box_aug_v3/val1/images/',
        pipeline=test_pipeline)
test_data3 = dict(
        classes=classes,
        type='Gd1024Dataset',
        ann_file='data/gd_newAug5_Rot0_4classes/box_aug_v3/val1/val1.json',
        img_prefix='data/gd_newAug5_Rot0_4classes/box_aug_v3/val1/images/',
        pipeline=test_pipeline)
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type='ConcatDataset',
        datasets=[train_data1, train_data2, train_data3]
    ),
    val=dict(
        type='ConcatDataset',
        datasets=[val_data1, val_data2, val_data3]
    ),
    test=dict(
        classes=classes,
        type='Gd1024Dataset',
        ann_file='data/gd_newAug5_Rot0_4classes/box_aug_v3/val1/val1.json',
        img_prefix='data/gd_newAug5_Rot0_4classes/box_aug_v3/val1/images/',
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='bbox')
