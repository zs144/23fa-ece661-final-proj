# import torch, torchvision
# import sys
# import mmseg
# import mmcv
# import mmengine
# import matplotlib.pyplot as plt
# import numpy as np
# import os.path as osp
# from PIL import Image
import os
from mmengine import Config
from utils import download_model, create_train_val_split, count_layers, count_frozen_layers, freeze_backbone_layers
from mmseg.registry import DATASETS
from mmseg.datasets import BaseSegDataset
from config_file import modify_config_file
from mmengine.runner import Runner
import argparse



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Transfer Learning')
    parser.add_argument('--fold', type=int, default=None, choices=[0,1,2,3,4])
    parser.add_argument('--freeze', action='store_true', default=False,   help='Freeze Backbone Layers' )
    args = parser.parse_args()

    data_name = 'duke'
    data_root = f'data/{data_name}'
    img_dir = 'images'
    ann_dir = 'labels'
    classes = ('background', 'fluid')
    palette =  [[0, 0, 0], [255,0,0]]
    pretrained = True
    fold = args.fold
    freeze = args.freeze

    # swin-tiny-patch4-window7-in1k-pre_upernet_8xb2-160k_ade20k-512x512
    # # mobilenet-v3-d8_lraspp_4xb4-320k_cityscapes-512x1024
    # deeplabv3plus_r50-d8_4xb2-40k_cityscapes-512x1024
    #unet-s5-d16_fcn_4xb4-160k_cityscapes-512x1024

    model_config = 'swin-tiny-patch4-window7-in1k-pre_upernet_8xb2-160k_ade20k-512x512'
    save_model_path = f'/NAS/user_data/user/ld258/output/work_dirs/{data_name}/' +model_config + '_freeze_' + str(freeze) + '_pretrained_' + str(pretrained) + '_dice_loss_' +  os.sep

    if not os.path.exists(save_model_path):
      os.makedirs(save_model_path)


    if not os.path.exists(save_model_path + model_config + '.py'):
      download_model(model_config, save_model_path)

    # create_train_val_split(data_root, ann_dir)
    original_cfg = Config.fromfile(save_model_path + model_config + '.py')
    cfg = modify_config_file(original_cfg, data_root, img_dir, ann_dir, save_model_path, fold, pretrained = pretrained )

    # print(f'Config:\n{cfg.pretty_text}')


    @DATASETS.register_module()
    class OCTDataset(BaseSegDataset):
      METAINFO = dict(classes = classes, palette = palette)
      def __init__(self, **kwargs):
        super().__init__(img_suffix='.png', seg_map_suffix='.png', **kwargs)


    runner = Runner.from_cfg(cfg)
    model = runner.model

    total_layers, backbone_layers, other_layers = count_layers(model)
    print(f"Total number of layers: {total_layers}")
    print(f"Number of layers in backbone: {backbone_layers}")
    print(f"Number of other layers: {other_layers}")

    if freeze:
      freeze_backbone_layers(model)
      frozen_layers_count = count_frozen_layers(model)
      print(f"Number of frozen layers: {frozen_layers_count}")


    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total trainable parameters: {total_params}")

    runner.train()
