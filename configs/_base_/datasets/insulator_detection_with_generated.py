# dataset settings
dataset_type = 'InsulatorDataset'
data_root = 'data/insulator/'
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
        type=dataset_type,
        classes=None,
        ann_file=data_root + 'train.json',
        img_prefix=data_root + 'images/',
        pipeline=train_pipeline)
val_data1 = dict(
        type=dataset_type,
        classes=None,
        ann_file=data_root + 'val.json',
        img_prefix=data_root + 'images/',
        pipeline=test_pipeline)
test_data1 = dict(
        type=dataset_type,
        classes=None,
        ann_file=data_root + 'val.json',
        img_prefix=data_root + 'images/',
        pipeline=test_pipeline)

train_data2 = dict(
        type=dataset_type,
        classes=('good', 'bad', 'defect'),
        ann_file='data/insulator_generated/train/train.json',
        img_prefix='data/insulator_generated/train/images/',
        pipeline=train_pipeline)
val_data2 = dict(
        type=dataset_type,
        classes=('good', 'bad', 'defect'),
        ann_file=data_root + 'data/insulator_generated/val/val.json',
        img_prefix=data_root + 'data/insulator_generated/val/images/',
        pipeline=test_pipeline)
test_data2 = dict(
        type=dataset_type,
        classes=('good', 'bad', 'defect'),
        ann_file=data_root + 'data/insulator_generated/val/val.json',
        img_prefix=data_root + 'data/insulator_generated/val/images/',
        pipeline=test_pipeline)


data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type='ConcatDataset',
        datasets=[train_data1, train_data2]
    ),
    val=dict(
        type='ConcatDataset',
        datasets=[val_data1, val_data2]
    ),
    test=dict(
        type='ConcatDataset',
        datasets=[test_data1, test_data2]
    ))
evaluation = dict(interval=1, metric='bbox')
