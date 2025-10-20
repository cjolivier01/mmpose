_base_ = ['../../../_base_/default_runtime.py']

# runtime
max_epochs = 10
base_lr = 4e-3

train_cfg = dict(max_epochs=max_epochs, val_interval=1)
randomness = dict(seed=21)

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=base_lr, weight_decay=0.05),
    paramwise_cfg=dict(norm_decay_mult=0, bias_decay_mult=0, bypass_duplicate=True))

# learning rate
param_scheduler = [
    dict(type='LinearLR', start_factor=1.0e-5, by_epoch=False, begin=0, end=1000),
    dict(
        type='CosineAnnealingLR', eta_min=base_lr * 0.05,
        begin=max_epochs // 2, end=max_epochs, T_max=max_epochs // 2,
        by_epoch=True, convert_to_iter_based=True),
]

# automatically scaling LR based on the actual training batch size
auto_scale_lr = dict(base_batch_size=256)

# codec settings
codec = dict(
    type='SimCCLabel',
    input_size=(288, 384),
    sigma=(6., 6.93),
    simcc_split_ratio=2.0,
    normalize=False,
    use_dark=False)

# model settings
model = dict(
    type='TopdownPoseEstimator',
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True),
    backbone=dict(
        _scope_='mmdet',
        type='CSPNeXt', arch='P5', expand_ratio=0.5,
        deepen_factor=1., widen_factor=1., out_indices=(4,),
        channel_attention=True, norm_cfg=dict(type='SyncBN'), act_cfg=dict(type='SiLU')),
    head=dict(
        type='RTMCCHead',
        in_channels=1024,
        out_channels=18,  # HARPE has 18 joints
        input_size=codec['input_size'],
        in_featuremap_size=tuple([s // 32 for s in codec['input_size']]),
        simcc_split_ratio=codec['simcc_split_ratio'],
        final_layer_kernel_size=7,
        gau_cfg=dict(
            hidden_dims=256,
            s=128,
            expansion_factor=2,
            dropout_rate=0.,
            drop_path=0.,
            act_fn='SiLU',
            use_rel_bias=False,
            pos_enc=False),
        loss=dict(
            type='KLDiscretLoss',
            use_target_weight=True,
            beta=10.,
            label_softmax=True),
        decoder=codec),
    test_cfg=dict(flip_test=True, ))

# initialize from full RTMPose-L SimCC pretrained checkpoint
load_from = 'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-l_simcc-aic-coco_pt-aic-coco_420e-384x288-97d6cb0f_20230228.pth'

# base dataset settings
dataset_type = 'HarpeDataset'
data_mode = 'topdown'
data_root = '/mnt/ripper-data2/datasets/VIP-HARPET/'

backend_args = dict(backend='local')

# pipelines
train_pipeline = [
    dict(type='LoadImage', backend_args=backend_args),
    dict(type='GetBBoxCenterScale'),
    dict(type='RandomFlip', direction='horizontal'),
    dict(type='RandomHalfBody'),
    dict(type='RandomBBoxTransform', scale_factor=[0.6, 1.4], rotate_factor=80),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='mmdet.YOLOXHSVRandomAug'),
    # Albumentations (optional). If not installed, skip this block.
    dict(type='GenerateTarget', encoder=codec),
    dict(type='PackPoseInputs')
]
val_pipeline = [
    dict(type='LoadImage', backend_args=backend_args),
    dict(type='GetBBoxCenterScale'),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='PackPoseInputs')
]

# data loaders
train_dataloader = dict(
    batch_size=64,
    num_workers=6,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        metainfo=dict(from_file='configs/_base_/datasets/harpe18.py'),
        data_root=data_root,
        data_mode=data_mode,
        ann_file='annot_train.h5',
        data_prefix=dict(img='images_train'),
        pipeline=train_pipeline,
        test_mode=False,
    ))
val_dataloader = dict(
    batch_size=64,
    num_workers=6,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type=dataset_type,
        metainfo=dict(from_file='configs/_base_/datasets/harpe18.py'),
        data_root=data_root,
        data_mode=data_mode,
        ann_file='annot_valid.h5',
        data_prefix=dict(img='images_valid'),
        test_mode=True,
        pipeline=val_pipeline,
    ))
test_dataloader = val_dataloader

# hooks
default_hooks = dict(
    checkpoint=dict(save_best='PCK', rule='greater', max_keep_ckpts=1))

# evaluators (PCK and AUC/EPE optional)
val_evaluator = [
    dict(type='PCKAccuracy', thr=0.2, norm_item='bbox'),
    dict(type='AUC'),
    dict(type='EPE'),
]
test_evaluator = val_evaluator
