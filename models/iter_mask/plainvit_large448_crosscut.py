from isegm.utils.exp_imports.default import *
from isegm.model.modeling.transformer_helper.cross_entropy_loss import CrossEntropyLoss

MODEL_NAME = 'plainvit_large448_crosscut'


def main(cfg):
    model, model_cfg = init_model(cfg)
    train(model, cfg, model_cfg)


def init_model(cfg):
    model_cfg = edict()
    model_cfg.crop_size = (896, 896)
    model_cfg.num_max_points = 24

    backbone_params = dict(
        img_size=(448, 448),
        patch_size=(16,16),
        in_chans=3,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4, 
        qkv_bias=True,
    )

    neck_params = dict(
        in_dim = 1024,
        out_dims = [192, 384, 768, 1536],
    )

    head_params = dict(
        in_channels=[192, 384, 768, 1536],
        in_index=[0, 1, 2, 3],
        dropout_ratio=0.1,
        num_classes=1,
        loss_decode=CrossEntropyLoss(),
        align_corners=False,
        upsample=cfg.upsample,
        channels={'x1': 256, 'x2': 128, 'x4': 64}[cfg.upsample],
    )


    model = PlainVitCrosscutModel(
        use_disks=True,
        norm_radius=5,
        with_prev_mask=True,
        backbone_params=backbone_params,
        neck_params=neck_params,
        head_params=head_params,
        random_split=cfg.random_split,
        slice_number=2
    )

    model.to(cfg.device)

    return model, model_cfg


def train(model, cfg, model_cfg):
    cfg.batch_size = 8 if cfg.batch_size < 1 else cfg.batch_size
    cfg.val_batch_size = cfg.batch_size
    crop_size = model_cfg.crop_size

    loss_cfg = edict()
    loss_cfg.instance_loss = NormalizedFocalLossSigmoid(alpha=0.5, gamma=2)
    loss_cfg.instance_loss_weight = 1.0

    train_augmentator = Compose([          
        UniformRandomResize(scale_range=(0.75, 1.25)),
        Flip(),
        RandomRotate90(),
        ShiftScaleRotate(shift_limit=0.03, scale_limit=0,
                         rotate_limit=(-3, 3), border_mode=0, p=0.75),
        PadIfNeeded(min_height=crop_size[0], min_width=crop_size[1], border_mode=0),
        RandomCrop(*crop_size),
        RandomBrightnessContrast(brightness_limit=(-0.25, 0.25), contrast_limit=(-0.15, 0.4), p=0.75),
        RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, p=0.75)
    ], p=1.0)

    val_augmentator = Compose([
        PadIfNeeded(min_height=crop_size[0], min_width=crop_size[1], border_mode=0),
        RandomCrop(*crop_size)
    ], p=1.0)

    points_sampler = MultiPointSampler(model_cfg.num_max_points, prob_gamma=0.80,
                                       merge_objects_prob=0.0,
                                       max_num_merged_objects=1)

    trainset = ProportionalComposeDataset([
        iSAIDDataset(
        cfg.ISAID_PATH,
        split='train',
        augmentator=train_augmentator,
        keep_background_prob=0.05,
        points_sampler=points_sampler,
    ),

        DeepglobeEvaluationDataset(
        cfg.DEEPGLOBE_PATH,
        split='train',
        augmentator=train_augmentator,
        keep_background_prob=0.05,
        points_sampler=points_sampler,
    ),

        InriaEvaluationDataset(
        cfg.INRIA_PATH,
        split='train',
        augmentator=train_augmentator,
        keep_background_prob=0.05,
        points_sampler=points_sampler,
    )
    ],
        ratios=[0.4,0.3,0.3],
        augmentator=train_augmentator,
        keep_background_prob=0.05,
        points_sampler=points_sampler,
        epoch_len=4500
    )

    valset = ProportionalComposeDataset([
        iSAIDDataset(
        cfg.ISAID_PATH,
        split='val',
        augmentator=val_augmentator,
        points_sampler=points_sampler,
    ),

        DeepglobeEvaluationDataset(
        cfg.DEEPGLOBE_PATH,
        split='val',
        augmentator=val_augmentator,
        points_sampler=points_sampler,
    ),

        InriaEvaluationDataset(
        cfg.INRIA_PATH,
        split='test',
        augmentator=val_augmentator,
        points_sampler=points_sampler,
    )
    ],
        ratios=[0.4,0.3,0.3],
        augmentator=val_augmentator,
        points_sampler=points_sampler,
        epoch_len=2000
    )


    optimizer_params = {
        'lr': 5e-5, 'betas': (0.9, 0.999), 'eps': 1e-8
    }

    lr_scheduler = partial(torch.optim.lr_scheduler.MultiStepLR,
                           milestones=[50, 55], gamma=0.1)
    trainer = ISTrainer(model, cfg, model_cfg, loss_cfg,
                        trainset, valset,
                        optimizer='adam',
                        optimizer_params=optimizer_params,
                        lr_scheduler=lr_scheduler,
                        checkpoint_interval=[(0, 5), (50, 1)],
                        image_dump_interval=300,
                        metrics=[AdaptiveIoU()],
                        max_interactive_points=model_cfg.num_max_points,
                        max_num_next_clicks=3)
    trainer.run(num_epochs=55, validation=False)
