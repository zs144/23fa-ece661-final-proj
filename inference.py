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
from utils import read_scalar_json, extract_data, find_max_mIoU_iterations

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

def plot_and_save_averages(all_data, metric_index, metric_name, filename, model_config):
    avg_freeze_on, avg_freeze_off = calculate_average_across_folds(all_data, metric_index)

    # Sorting the keys to ensure the line plots are in order
    iterations_on, averages_on = zip(*sorted(avg_freeze_on.items()))
    iterations_off, averages_off = zip(*sorted(avg_freeze_off.items()))

    plt.figure()
    plt.plot(iterations_on, averages_on, label='Freeze ON', linestyle='-')
    plt.plot(iterations_off, averages_off, label='Freeze OFF', linestyle='--')

    plt.xlabel('Iteration')
    plt.ylabel(metric_name)
    plt.title(f'Average {metric_name} for {model_config}')
    plt.legend()

    # Set y-axis limits for mIoU and mAcc if necessary
    if metric_name in ['mIoU', 'mAcc']:
        plt.ylim(0, 100)

    plt.savefig(filename)


def calculate_average_across_folds(all_data, metric_index):
    sums_freeze_on, counts_freeze_on = defaultdict(float), defaultdict(int)
    sums_freeze_off, counts_freeze_off = defaultdict(float), defaultdict(int)

    # Accumulate sums and counts for each iteration
    for fold_data in all_data.values():
        for freeze, metrics in fold_data.items():
            for x, y in metrics[metric_index]:
                if freeze:
                    sums_freeze_on[x] += y
                    counts_freeze_on[x] += 1
                else:
                    sums_freeze_off[x] += y
                    counts_freeze_off[x] += 1

    # Calculate averages
    avg_freeze_on = {x: sums_freeze_on[x] / counts_freeze_on[x] for x in sums_freeze_on}
    avg_freeze_off = {x: sums_freeze_off[x] / counts_freeze_off[x] for x in sums_freeze_off}

    return avg_freeze_on, avg_freeze_off



def plot_and_save_all_folds(results, metric_index, metric_name, filename):
    plt.figure()
    for fold, data in results.items():
        x, y = zip(*data[metric_index])
        plt.plot(x, y, label=f'Fold {fold}')

    plt.xlabel('Iteration')
    plt.ylabel(metric_name)
    plt.title(f'{metric_name} vs Iteration')
    plt.legend()
    if metric_name in ['mIoU', 'mAcc']:
        plt.ylim(0, 100)

    # if metric_name in ['mAcc']:
    #     plt.ylim(0, 100)

    plt.savefig(filename)


def plot_and_save_metric(all_data, metric_index, metric_name, filename):
    plt.figure()

    # Iterate through each fold and freeze status
    for fold, freeze_data in all_data.items():
        for freeze, metrics in freeze_data.items():
            x, y = zip(*metrics[metric_index])
            label = f'Fold {fold} - Freeze {"ON" if freeze else "OFF"}'
            linestyle = '-' if freeze else '--'
            plt.plot(x, y, label=label, linestyle=linestyle)

    plt.xlabel('Iteration')
    plt.ylabel(metric_name)
    plt.title(f'{metric_name} across Folds with Freeze Status')
    plt.legend()

    # Set y-axis limits for mIoU and mAcc if necessary
    # if metric_name in ['mIoU', 'mAcc']:
    #     plt.ylim(0, 100)

    plt.savefig(filename)





def plot_comparison(ground_truth, prediction, save_path, fname):
    # Define conditions
    TP = (ground_truth == 1) & (prediction == 1)  # True Positive
    FP = (ground_truth == 0) & (prediction == 1)  # False Positive
    TN = (ground_truth == 0) & (prediction == 0)  # True Negative
    FN = (ground_truth == 1) & (prediction == 0)  # False Negative

    # Create an RGB image
    comparison_image = np.zeros(ground_truth.shape + (3,), dtype=int)
    comparison_image[TP] = [0, 255, 0]    # Green for TP
    comparison_image[FP] = [255, 0, 0]    # Red for FP
    comparison_image[TN] = [0, 0, 0]    # Blue for TN
    comparison_image[FN] = [255, 255, 0]  # Yellow for FN

    image = Image.fromarray(comparison_image.astype(np.uint8))

    image.save(save_path+ fname+'.png')


freeze_list = [ True, False]
data_root = 'data'
img_dir = 'images'
ann_dir = 'labels'
classes = ('background', 'fluid')
palette =  [[0, 0, 0], [255,0,0]]
fold_list = [0, 1, 2, 3, 4]
pretrained = False


model_config = 'mobilenet-v3-d8_lraspp_4xb4-320k_cityscapes-512x1024'

if model_config == 'deeplabv3plus_r50-d8_4xb2-40k_cityscapes-512x1024':
    preferred_name = 'DeepLabv3Plus'
if model_config == 'mobilenet-v3-d8_lraspp_4xb4-320k_cityscapes-512x1024':
    preferred_name = 'MobileNetV3'
if model_config == 'swin-tiny-patch4-window7-in1k-pre_upernet_8xb2-160k_ade20k-512x512':
    preferred_name = 'Swin Tiny'

base_path = '/NAS/user_data/user/ld258/output/work_dirs/'


all_data = {}

for freeze in freeze_list:

    load_model_path = f'{base_path}{model_config}_freeze_{freeze}_pretrained_{pretrained}/'

    original_cfg = Config.fromfile(load_model_path + model_config + '.py')
    cfg = modify_config_file(original_cfg, data_root, img_dir, ann_dir, load_model_path, 0 )
    for fold in fold_list:
        test_img_path = f'/codebase/data/splits/val_fold_{fold}.txt'
        fnames = open(test_img_path).read().splitlines()

        json_path  = f'{base_path}{model_config}_freeze_{freeze}_pretrained_{pretrained}/fold_{fold}_default_augmentation/'
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


for freeze in freeze_list:
    for fold in fold_list:

        iteration_number = max_mIoU_iters[fold][freeze]

        checkpoint_path = f'{base_path}{model_config}_freeze_{freeze}_pretrained_{pretrained}/fold_{fold}_default_augmentation/iter_{iteration_number}.pth'

        print (checkpoint_path)



        model = init_model(cfg, checkpoint_path, 'cuda:0')
        pred_path =  f'/codebase/work_dirs/pred/{model_config}/freeze_{freeze}_pretrained_{pretrained}/fold_{fold}/'
        overlay_path = f'/codebase/work_dirs/overlay/{model_config}/freeze_{freeze}_pretrained_{pretrained}/fold_{fold}/'
        auc_path  = f'/codebase/work_dirs/{model_config}/'
        if not os.path.exists(auc_path):
            os.makedirs(auc_path)


        if not os.path.exists(pred_path):
            os.makedirs(pred_path)

        if not os.path.exists(overlay_path):
            os.makedirs(overlay_path)

        fpath = '/codebase/data/'
        ext = '.png'
        threshold = 0.3

        all_gt = []
        all_preds = []


        for fname in fnames:
            img = mmcv.imread(fpath+'images/'+ fname + ext)
            gt =  mmcv.imread(fpath+'labels/'+ fname + ext)
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
            img = Image.fromarray(seg[0,:,:].astype(bool))
            img.save(pred_path+ fname+ext)

            plot_comparison(np.array(gt[:,:,0]).astype(bool), seg[0,:,:].astype(bool), overlay_path, fname)



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