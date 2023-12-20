from config_file import modify_config_file
from mmengine import Config
from utils import read_scalar_json, extract_data, find_max_mIoU_iterations
import os
from mmseg.apis import init_model, inference_model
import mmcv
from config_file import modify_config_file
from mmengine import Config

import numpy as np
from PIL import Image

freeze_list = [False, True]
data_root = 'data'
img_dir = 'images'
ann_dir = 'labels'
classes = ('background', 'fluid')
palette =  [[0, 0, 0], [255,0,0]]
fold_list = [0, 1, 2, 3, 4]


model_config = 'swin-tiny-patch4-window7-in1k-pre_upernet_8xb2-160k_ade20k-512x512'

if model_config == 'deeplabv3plus_r50-d8_4xb2-40k_cityscapes-512x1024':
    preferred_name = 'deeplabv3plus pretrained cityscapes'
if model_config == 'mobilenet-v3-d8_lraspp_4xb4-320k_cityscapes-512x1024':
    preferred_name = 'mobilenet-v3 pretrained cityscapes'
if model_config == 'swin-tiny-patch4-window7-in1k-pre_upernet_8xb2-160k_ade20k-512x512':
    preferred_name = 'swin pretrained cityscapes'

base_path = '/NAS/user_data/user/ld258/output/work_dirs/'


all_data = {}

for freeze in freeze_list:

    load_model_path = f'{base_path}{model_config}_freeze_{freeze}/'

    original_cfg = Config.fromfile(load_model_path + model_config + '.py')
    cfg = modify_config_file(original_cfg, data_root, img_dir, ann_dir, load_model_path, 0 )
    for fold in fold_list:
        test_img_path = f'/codebase/data/splits/val_fold_{fold}.txt'
        fnames = open(test_img_path).read().splitlines()

        json_path  = f'{base_path}{model_config}_freeze_{freeze}/fold_{fold}_default_augmentation/'
        json_data = read_scalar_json(json_path)
        iter_loss, iter_mIoU, iter_mAcc = extract_data(json_data)
        if fold not in all_data:
            all_data[fold] = {}

        all_data[fold][freeze] = (iter_loss, iter_mIoU, iter_mAcc)

# Assuming 'all_data' is your nested dictionary containing the data
max_mIoU_iters = find_max_mIoU_iterations(all_data)

for fold, data in max_mIoU_iters.items():
    for freeze, iteration in data.items():
        print(f"Fold {fold}, Freeze {freeze}: Max mIoU at iteration {iteration}")



test_base_path =  f'/codebase/test_data/'
test_path = f'/codebase/test_data/test_fnames.txt'
test_fnames = open(test_path).read().splitlines()


for freeze in freeze_list:
    for fold in fold_list:

        iteration_number = max_mIoU_iters[fold][freeze]

        checkpoint_path = f'{base_path}{model_config}_freeze_{freeze}/fold_{fold}_default_augmentation/iter_{iteration_number}.pth'

        print (checkpoint_path)

        model = init_model(cfg, checkpoint_path, 'cuda:0')
        pred_path =  f'/codebase/work_dirs/new_test_v2/pred/{model_config}/freeze_{freeze}/fold_{fold}/'
        overlay_path = f'/codebase/work_dirs/new_test_v2/overlay/{model_config}/freeze_{freeze}/fold_{fold}/'
        comparison_path = f'/codebase/work_dirs/new_test_v2/comparison/{model_config}/freeze_{freeze}/fold_{fold}/'
        if not os.path.exists(pred_path):
            os.makedirs(pred_path)

        if not os.path.exists(overlay_path):
            os.makedirs(overlay_path)

        if not os.path.exists(comparison_path):
            os.makedirs(comparison_path)



        fpath = '/codebase/test_data/'
        ext = '.png'
        threshold = 0.8

        for fname in test_fnames:
            img = mmcv.imread(test_base_path+'images/'+ fname + ext)
            print ('currently processing', fname)
            result = inference_model(model, img)

            seg_result = result.seg_logits.data
            seg_array = np.array(seg_result.cpu())
            seg = seg_array>threshold
            seg_img = Image.fromarray(seg[0,:,:].astype('uint8') * 255)  # Convert boolean mask to uint8

            # Convert original image to RGBA
            img_rgba = Image.fromarray(img).convert("RGBA")

            # Create a red mask where segmentation is True
            mask_color = Image.new("RGBA", seg_img.size, (0, 0, 0, 0))  # Start with a fully transparent image
            mask_red = Image.new("RGBA", seg_img.size, (255, 0, 0, 125))  # Red color with alpha
            mask_color.paste(mask_red, (0,0), mask=seg_img)  # Apply the red mask where seg_img is not 0

            # Blend the original image and the mask
            result_image = Image.alpha_composite(img_rgba, mask_color)
            result_image = result_image.convert("RGB")  # Convert back to RGB to save in JPEG or similar formats

            # Save the results
            seg_img.save(pred_path + fname + ext)
            result_image.save(overlay_path + fname + ext)

            img_pil = Image.fromarray(img)
            total_width = img_pil.width + result_image.width
            comparison_image = Image.new('RGB', (total_width, img_pil.height))
            comparison_image.paste(img_pil, (0, 0))
            comparison_image.paste(result_image, (img_pil.width, 0))

            # Save the comparison image
            comparison_image.save(comparison_path + fname + ext)



