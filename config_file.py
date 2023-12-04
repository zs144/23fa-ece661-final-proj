
# img_norm_cfg = dict(
#     mean=[123.675, 116.28, 103.53],  # Mean of the dataset
#     std=[58.395, 57.12, 57.375],     # Standard deviation of the dataset
#     to_rgb=True
# )
import os
import glob


def find_pth_files(path):
    # Check if the path exists
    if not os.path.exists(path):
        raise ValueError(f"The provided path does not exist: {path}")

    # Find all .pth files in the specified path
    pth_files = glob.glob(os.path.join(path, '*.pth'))

    # Check if any .pth files were found
    if not pth_files:
        raise FileNotFoundError("No .pth files found in the specified path.")

    # Return the list of full path names
    return pth_files

def modify_config_file(cfg, data_root, img_dir, ann_dir, save_model_path):
    cfg.crop_size = (512, 512)
    cfg.model.data_preprocessor.size = cfg.crop_size
    # cfg.model.backbone.norm_cfg =  cfg.norm_cfg
    # cfg.model.decode_head.norm_cfg =  cfg.norm_cfg
    # cfg.model.auxiliary_head.norm_cfg = cfg.norm_cfg
    # modify num classes of the model in decode/auxiliary head
    cfg.model.decode_head.num_classes = 1


    # cfg.model.auxiliary_head.num_classes =1
    cfg.model.decode_head.out_channels = 1
    cfg.model.decode_head.loss_decode.use_sigmoid=True

    # Modify dataset type and path
    cfg.dataset_type = 'OCTDataset'
    cfg.data_root = data_root

    cfg.train_dataloader.batch_size = 8

    cfg.train_pipeline = [
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations'),
        dict(type='RandomResize', scale=(512, 512), ratio_range=(0.5, 2.0), keep_ratio=True),
        dict(type='RandomCrop', crop_size=(512, 512), cat_max_ratio=0.75),
        # dict(type='PhotoMetricDistortion'),
        # dict(type='CLAHE'),
        # dict(type='Normalize', **img_norm_cfg),
        dict(type='RandomFlip', prob=0.5),
        dict(type='PackSegInputs')
    ]

    cfg.test_pipeline = [
        dict(type='LoadImageFromFile'),
        # dict(type='Resize', scale=(512, 512), keep_ratio=True),
        # add loading annotation after ``Resize`` because ground truth
        # does not need to do resize data transform
        dict(type='LoadAnnotations'),
        # dict(type='CLAHE'),
        # dict(type='Normalize', **img_norm_cfg),
        dict(type='PackSegInputs')
    ]


    cfg.train_dataloader.dataset.type = cfg.dataset_type
    cfg.train_dataloader.dataset.data_root = cfg.data_root
    cfg.train_dataloader.dataset.data_prefix = dict(img_path=img_dir, seg_map_path=ann_dir)
    cfg.train_dataloader.dataset.pipeline = cfg.train_pipeline
    cfg.train_dataloader.dataset.ann_file = 'splits/train.txt'

    cfg.val_dataloader.dataset.type = cfg.dataset_type
    cfg.val_dataloader.dataset.data_root = cfg.data_root
    cfg.val_dataloader.dataset.data_prefix = dict(img_path=img_dir, seg_map_path=ann_dir)
    cfg.val_dataloader.dataset.pipeline = cfg.test_pipeline
    cfg.val_dataloader.dataset.ann_file = 'splits/val.txt'

    cfg.test_dataloader = cfg.val_dataloader


    # Load the pretrained weights
    try:
        pth_files = find_pth_files(save_model_path)
        for file in pth_files:
            cfg.load_from = file
    except Exception as e:
        print(e)

    # Set up working dir to save files and logs.
    cfg.work_dir = save_model_path + 'default_augmentation/'
    if not os.path.exists(save_model_path + 'default_augmentation/'):
        os.makedirs(save_model_path + 'default_augmentation/')

    cfg.train_cfg.max_iters = 100000
    cfg.train_cfg.val_interval = 1000
    cfg.default_hooks.logger.interval = 1000
    cfg.default_hooks.checkpoint.interval = 1000

    # Set seed to facilitate reproducing the result
    cfg['randomness'] = dict(seed=0)

    return cfg