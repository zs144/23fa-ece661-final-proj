import subprocess
import mmengine
import os.path as osp

import torch.nn as nn
import os.path as osp
from sklearn.model_selection import KFold
import mmengine
from collections import defaultdict
import numpy as np
from PIL import Image
from sklearn.metrics import precision_recall_curve, roc_curve, auc
import json
import os
import matplotlib.pyplot as plt


def create_train_val_split(data_root, ann_dir, n_splits=5):
    # Split directory
    split_dir = 'splits'
    mmengine.mkdir_or_exist(osp.join(data_root, split_dir))

    # Read filenames and group by patient
    filename_list = [osp.splitext(filename)[0] for filename in mmengine.scandir(
        osp.join(data_root, ann_dir), suffix='.png')]
    patient_dict = {}
    for filename in filename_list:
        patient_id = filename.split('_')[1]  # Extract patient identifier
        patient_dict.setdefault(patient_id, []).append(filename)

    # Create list of patient groups
    patient_groups = list(patient_dict.values())

    # K-Fold cross-validation
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    for i, (train_idx, val_idx) in enumerate(kf.split(patient_groups)):
        # Get train and validation filenames
        train_files = [patient_groups[idx] for idx in train_idx]
        val_files = [patient_groups[idx] for idx in val_idx]

        # Flatten the lists
        train_files = [item for sublist in train_files for item in sublist]
        val_files = [item for sublist in val_files for item in sublist]

        # Write to files
        with open(osp.join(data_root, split_dir, f'train_fold_{i}.txt'), 'w') as f:
            f.writelines(line + '\n' for line in train_files)
        with open(osp.join(data_root, split_dir, f'val_fold_{i}.txt'), 'w') as f:
            f.writelines(line + '\n' for line in val_files)



def download_model(config_name, destination_folder):
    try:
        subprocess.run(['mim', 'download', 'mmsegmentation', '--config', config_name, '--dest', destination_folder], check=True)
        print(f"Model {config_name} downloaded successfully to {destination_folder}")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e}")


def count_layers(model):
    total_layers = 0
    backbone_layers = 0
    other_layers = 0

    for name, module in model.named_modules():
        if isinstance(module, nn.Module) and not isinstance(module, nn.Sequential) and not isinstance(module, nn.ModuleList) and module != model:
            total_layers += 1
            if 'backbone' in name:
                backbone_layers += 1
            else:
                other_layers += 1

    return total_layers, backbone_layers, other_layers


def count_frozen_layers(model):
    frozen_layers = 0

    for name, module in model.named_modules():
        # Check if the module is a high-level layer (e.g., Conv2d, Linear)
        if isinstance(module, nn.Module) and not isinstance(module, nn.Sequential) and not isinstance(module, nn.ModuleList) and module != model:
            all_params_frozen = all(not param.requires_grad for param in module.parameters())
            if all_params_frozen:
                frozen_layers += 1

    return frozen_layers

def freeze_backbone_layers(model):
    for name, param in model.named_parameters():
        if 'backbone' in name:  # Adjust this condition as needed
            # print ('We are freezing Layers')
            param.requires_grad = False




def read_scalar_json(root_dir='.'):
    for root, dirs, files in os.walk(root_dir):
        if 'scalars.json' in files:
            file_path = os.path.join(root, 'scalars.json')
            try:
                json_objects = []
                with open(file_path, 'r') as file:
                    for line in file:
                        if line.strip():  # Check if line is not empty
                            json_object = json.loads(line)
                            json_objects.append(json_object)
                return json_objects
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON in {file_path}: {e}")
            except Exception as e:
                print(f"Error reading file {file_path}: {e}")

    return None

def extract_data(json_objects):
    iter_loss, iter_mIoU, iter_mAcc = [], [], []
    for obj in json_objects:

        if 'loss' in obj :
            iter_loss.append((obj['iter'], obj['loss']))
        if 'mIoU' in obj and 'mAcc' in obj:
            iter_mIoU.append((obj['step'], obj['mIoU']))
            iter_mAcc.append((obj['step'], obj['mAcc']))
    return iter_loss, iter_mIoU, iter_mAcc


def find_max_mIoU_iterations(all_data):
    max_mIoU_iterations = {}

    for fold, freeze_data in all_data.items():
        max_mIoU_iterations[fold] = {}
        for freeze, metrics in freeze_data.items():
            iter_mIoU = metrics[1]  # Assuming index 1 is for iter_mIoU
            max_mIoU = max(iter_mIoU, key=lambda x: x[1])  # Find the tuple with the max mIoU value
            max_mIoU_iterations[fold][freeze] = max_mIoU[0]  # Store the iteration number

    return max_mIoU_iterations





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


def dice_score(y_true, y_pred):
    eps = 1e-8
    intersection = np.sum(y_true * y_pred)

    # Check if both y_true and y_pred are all zeros
    if np.sum(y_true) == 0 and np.sum(y_pred) == 0:
        return 1.0
    else:
        return (2. * intersection) / (np.sum(y_true) + np.sum(y_pred) + eps)



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