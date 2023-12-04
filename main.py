import torch, torchvision
import mmseg
import mmcv
import mmengine
import matplotlib.pyplot as plt
import numpy as np
import os.path as osp
from PIL import Image
import os
from mmengine import Config
from utils import download_model, create_train_val_split

from mmseg.registry import DATASETS
from mmseg.datasets import BaseSegDataset

from config_file import modify_config_file
from mmengine.runner import Runner


data_root = 'data'
img_dir = 'images_gaussian'
ann_dir = 'labels'
classes = ('background', 'fluid')
palette =  [[0, 0, 0], [255,0,0]]

# swin-tiny-patch4-window7-in1k-pre_upernet_8xb2-160k_ade20k-512x512
model_config = 'deeplabv3plus_r50-d8_4xb2-40k_cityscapes-512x1024'
# mobilenet-v3-d8_lraspp_4xb4-320k_cityscapes-512x1024

save_model_path = '/codebase/work_dirs/' +model_config+ '_image_preprocessed'+ os.sep
if not os.path.exists(save_model_path+model_config):
  os.makedirs(save_model_path+model_config)


download_model(model_config, save_model_path)
create_train_val_split(data_root, ann_dir)
original_cfg = Config.fromfile('/codebase/work_dirs/' + model_config+ os.sep + model_config + '.py')
cfg = modify_config_file(original_cfg, data_root, img_dir, ann_dir, save_model_path )
# print(f'Config:\n{cfg.pretty_text}')


@DATASETS.register_module()
class OCTDataset(BaseSegDataset):
  METAINFO = dict(classes = classes, palette = palette)
  def __init__(self, **kwargs):
    super().__init__(img_suffix='.png', seg_map_suffix='.png', **kwargs)




runner = Runner.from_cfg(cfg)

runner.train()
