import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import seaborn as sns

def dice_score(y_true, y_pred):
    eps = 1e-8
    intersection = np.sum(y_true * y_pred)

    # Check if both y_true and y_pred are all zeros
    if np.sum(y_true) == 0 and np.sum(y_pred) == 0:
        return 1.0
    else:
        return (2. * intersection) / (np.sum(y_true) + np.sum(y_pred) + eps)

def compute_dice_for_folder(pred_folder_path, gt_folder_path):
    dice_scores = []
    for filename in os.listdir(pred_folder_path):
        if filename.endswith('.png'):
            pred_path = os.path.join(pred_folder_path, filename)
            gt_path = os.path.join(gt_folder_path, filename)  # Corresponding ground truth file
            # print ('pre dfull path' , pred_path, gt_path)
            if os.path.exists(gt_path):  # Check if ground truth file exists
                # Load prediction and ground truth
                pred = np.array(Image.open(pred_path))  # Convert to binary mask
                gt = np.array(Image.open(gt_path))  # Convert to binary mask

                if len(np.unique(gt))>1:
                # print ('shape of pred and gt', pred.shape, gt.shape)
                # Compute Dice score
                    score = dice_score(gt.flatten(), pred.flatten())

                    dice_scores.append(score)

    return np.mean(dice_scores)  # Return the average Dice score

# Updated code to compute Dice scores and generate the plot

# Define the base directories for Duke and UMN datasets


datasets = ['Duke', 'UMN']

base_dir = 'D:\\projects\\phd_assignments\\ece661\\project\\oct_segmentation\\work_dirs\\comparison_duke_and_umn\\'
methods = ['aug_gaussian_blur', 'aug_simple', 'loss_dice_and_cross_entropy']
folds = ['fold_0', 'fold_1']

# results = {dataset: {method: {} for method in methods} for dataset in datasets}
data = []

# Compute Dice scores for each method and fold
for dataset in datasets:
    for method in methods:
        for fold in folds:
            pred_folder_path = os.path.join(base_dir, dataset, method, 'pred', fold)
            gt_folder_path = os.path.join(base_dir, dataset, 'labels')
             # Assuming ground truth is in a 'groundtruth' folder
            avg_dice = compute_dice_for_folder(pred_folder_path, gt_folder_path)
            # print (dataset, method, fold, avg_dice)
            data.append([dataset, method, fold, avg_dice])

            # results[dataset][method][fold] = avg_dice  # Store the result in the dictionary

df = pd.DataFrame(data, columns=['Dataset', 'Method', 'Fold', 'Dice Score'])


# # Save the figure
# plt.savefig('D:\\projects\\phd_assignments\\ece661\\project\\oct_segmentation\\work_dirs\\comparison_duke_and_umn\\dice_scores_plot.png')
# plt.close()

# Creating a bar plot that includes results from both datasets for fold 1 using seaborn

# Filtering the DataFrame for fold 1
df_fold1 = df[df['Fold'] == 'fold_1']

# Setting the figure size
plt.figure(figsize=(12, 7))

# Creating the bar plot
bar_plot = sns.barplot(data=df_fold1, x='Dataset', y='Dice Score', hue='Method')

# Adding labels and title
plt.title('Comparison of Dice Scores for test set Across Datasets and different methods')
plt.ylabel('Dice Score')
plt.xlabel('Dataset')
for p in bar_plot.patches:
    bar_plot.annotate(format(p.get_height(), '.2f'),
                      (p.get_x() + p.get_width() / 2., p.get_height()),
                      ha='center', va='center',
                      xytext=(0, 9),
                      textcoords='offset points')
# Displaying the plot
save_path = f'D:/projects/phd_assignments/ece661/project/oct_segmentation/work_dirs/comparison_duke_and_umn/dice_plot_no_zero_background.svg'
plt.savefig(save_path, format='svg')
plt.show()


