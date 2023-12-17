import os
import matplotlib.pyplot as plt
import mmcv
from mmseg.apis import init_model, inference_model, show_result_pyplot
from config_file import modify_config_file
from mmengine import Config
from collections import defaultdict
import numpy as np
from PIL import Image
from sklearn.metrics import precision_recall_curve, roc_curve, auc
from utils import *

# def plot_and_save(data, x_label, y_label, title, filename):
#     if not data:
#         print(f"No data available for {title}. Skipping plot.")
#         return
#     x, y = zip(*data)
#     plt.figure()
#     plt.plot(x, y)
#     plt.xlabel(x_label)
#     plt.ylabel(y_label)
#     plt.title(title)
#     plt.savefig(filename)



freeze_list = [ False]
data_root = 'data/UMN'
img_dir = 'images_umn'
ann_dir = 'labels_umn'
classes = ('background', 'fluid')
palette =  [[0, 0, 0], [255,0,0]]
fold_list = [0]
pretrained = True
aug_type = '_baseline_'
dataset = 'UMN_data'

model_config = 'swin-tiny-patch4-window7-in1k-pre_upernet_8xb2-160k_ade20k-512x512'

if model_config == 'deeplabv3plus_r50-d8_4xb2-40k_cityscapes-512x1024':
    preferred_name = 'DeepLabv3Plus'
if model_config == 'mobilenet-v3-d8_lraspp_4xb4-320k_cityscapes-512x1024':
    preferred_name = 'MobileNetV3'
if model_config == 'swin-tiny-patch4-window7-in1k-pre_upernet_8xb2-160k_ade20k-512x512':
    preferred_name = 'Swin Tiny'

base_path = f'/NAS/user_data/user/ld258/output/work_dirs/{dataset}/'

all_data = {}
dice_scores = []

for freeze in freeze_list:

    load_model_path = f'{base_path}{model_config}_freeze_{freeze}_pretrained_{pretrained}{aug_type}/'
    print ('Loading MOdel Pathts ', load_model_path)

    original_cfg = Config.fromfile(load_model_path + model_config + '.py')
    cfg = modify_config_file(original_cfg, data_root, img_dir, ann_dir, load_model_path, 0, pretrained )

    for fold in fold_list:
        test_img_path = f'/codebase/{data_root}/splits/val_fold_{fold}.txt'
        fnames = open(test_img_path).read().splitlines()

        json_path  = f'{base_path}{model_config}_freeze_{freeze}_pretrained_{pretrained}{aug_type}/fold_{fold}_default_augmentation/'
        json_data = read_scalar_json(json_path)
        iter_loss, iter_mIoU, iter_mAcc = extract_data(json_data)
        if fold not in all_data:
            all_data[fold] = {}

        all_data[fold][freeze] = (iter_loss, iter_mIoU, iter_mAcc)


# plot_and_save_averages(all_data, 1, 'mIoU', f'{model_config}_average_mIoU_vs_iter.png', preferred_name)


# Assuming 'all_data' is your nested dictionary containing the data
max_mIoU_iters = find_max_mIoU_iterations(all_data)

for fold, data in max_mIoU_iters.items():
    for freeze, iteration in data.items():
        print(f"Fold {fold}, Freeze {freeze}: Max mIoU at iteration {iteration}")



fold_list = [0,1 ]
for freeze in freeze_list:
    for fold in fold_list:

        iteration_number = max_mIoU_iters[0][freeze]

        checkpoint_path = f'{base_path}{model_config}_freeze_{freeze}_pretrained_{pretrained}{aug_type}/fold_{0}_default_augmentation/iter_{iteration_number}.pth'

        print (checkpoint_path)


        model = init_model(cfg, checkpoint_path, 'cuda:0')
        pred_path =  f'/codebase/work_dirs_new/{data_root}/{aug_type}/pred/{model_config}/freeze_{freeze}_pretrained_{pretrained}/fold_{fold}/'
        overlay_path = f'/codebase/work_dirs_new/{data_root}/{aug_type}/overlay/{model_config}/freeze_{freeze}_pretrained_{pretrained}/fold_{fold}/'
        auc_path  = f'/codebase/work_dirs_new/{data_root}/{aug_type}/{model_config}/'
        if not os.path.exists(auc_path):
            os.makedirs(auc_path)


        if not os.path.exists(pred_path):
            os.makedirs(pred_path)

        if not os.path.exists(overlay_path):
            os.makedirs(overlay_path)

        fpath = f'/codebase/{data_root}/'
        ext = '.png'
        threshold = 0.3

        all_gt = []
        all_preds = []

        test_img_path = f'/codebase/{data_root}/splits/val_fold_{fold}.txt'
        fnames = open(test_img_path).read().splitlines()
        fold_dice_scores = []
        for fname in fnames:
            img = mmcv.imread(fpath+'images_umn/'+ fname + ext)
            gt =  mmcv.imread(fpath+'labels_umn/'+ fname + ext)
            print ('currently processing', fname)
            result = inference_model(model, img)


            seg_result = result.seg_logits.data
            seg_array = np.array(seg_result.cpu())

            seg_pred = seg_array.squeeze().flatten()

            # Flatten the ground truth
            gt_flat = gt[:,:,0].astype(bool).flatten()

            all_gt.extend(gt_flat)
            all_preds.extend(seg_pred)



            seg = seg_array>threshold

            dice = dice_score(gt_flat, seg_pred>threshold)
            fold_dice_scores.append(dice)


            img = Image.fromarray(seg[0,:,:].astype(bool))



            img.save(pred_path+ fname+ext)

            plot_comparison(np.array(gt[:,:,0]).astype(bool), seg[0,:,:].astype(bool), overlay_path, fname)


        dice_scores.append(np.mean(fold_dice_scores))
        # Compute Precision-Recall for this fold
        precision, recall, _ = precision_recall_curve(all_gt, all_preds)
        pr_auc = auc(recall, precision)

    # Plot Precision-Recall curve for this fold
        plt.plot(recall, precision, lw=2, label=f'Fold {fold} (AUC = {pr_auc:.2f})')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'{preferred_name} Pretrained {pretrained} Freeze {freeze}')
    plt.legend(loc="lower left")
    plt.savefig(f'{auc_path}/{preferred_name}_freeze_{freeze}_pretrained_{pretrained}_pr_curve.png')
    plt.close()

    plt.figure()
    plt.bar(range(len(dice_scores)), dice_scores, tick_label=[f'Fold {i}' for i in fold_list])
    plt.xlabel('Fold')
    plt.ylabel('Average Dice Score')
    plt.title(f'{preferred_name} Pretrained {pretrained} Freeze {freeze} - Dice Scores')
    plt.savefig(f'{auc_path}/{preferred_name}_freeze_{freeze}_pretrained_{pretrained}_dice_scores.png')
    plt.close()



# Assuming 'results' is your dictionary containing the data
# Plot and save 'iter_loss' for all folds
# plot_and_save_all_folds(all_data, 0, 'Loss',  f'{model_config}_freeze_{freeze}_loss_vs_iter.png')

# # Plot and save 'iter_mIoU' for all folds
# plot_and_save_all_folds(all_data, 1, 'mIoU', f'{model_config}_freeze_{freeze}_mIoU_vs_iter.png')

# # Plot and save 'iter_mAcc' for all folds
# plot_and_save_all_folds(all_data, 2, 'mAcc', f'{model_config}_freeze_{freeze}_mAcc_vs_iter.png')




    # plt.figure(figsize=(8, 6))
    # vis_result = show_result_pyplot(model, gt, result)
    # # plt.imshow(mmcv.bgr2rgb(vis_result))
    # plt.savefig(pred_path+fname+ext)



# checkpoint_path = './work_dirs/tutorial/iter_6800.pth'
# model = init_model(cfg, checkpoint_path, 'cuda:0')
# pred_path =  '/codebase/work_dirs/results/'
# fpath = '/codebase/data/'
# ext = '.png'
# some_vis = '/codebase/work_dirs/results/'
# threshold = 0.3

# for fname in fnames:
#     img = mmcv.imread(fpath+'images/'+ fname + ext)
#     gt =  mmcv.imread(fpath+'labels/'+ fname + ext)

#     result = inference_model(model, img)
#     plt.figure(figsize=(8, 6))
#     vis_result = show_result_pyplot(model, gt, result)
#     # plt.imshow(mmcv.bgr2rgb(vis_result))
#     plt.savefig(some_vis+fname+ext)


# plot_and_save(iter_loss, 'Iteration', 'Loss', 'Loss vs Iteration', f'{model_config}_freeze_{freeze}_loss_vs_iter.png')
# plot_and_save(iter_mIoU, 'Iteration', 'mIoU', 'mIoU vs Iteration', f'{model_config}_freeze_{freeze}_mIoU_vs_iter.png')

# # Plot and save 'mAcc vs iter'
# plot_and_save(iter_mAcc, 'Iteration', 'mAcc', 'mAcc vs Iteration', f'{model_config}_freeze_{freeze}_mAcc_vs_iter.png')