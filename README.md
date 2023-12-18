# Intra-retinal cyst segmentation in optical coherence tomography images via transfer learning

This repository contains the source code and related materials for the ECE 661 final project by Lavsen Dahal, Ken Ho, and Zion Sheng.

## Quick Navigation

| Files/Directory         | Comments                                                     |
| ----------------------- | ------------------------------------------------------------ |
| `config_file.py`        | Set up the training/testing configuration.                   |
| `convert_mat_to_png.py` | Convert original `.mat` files to `.png` files.               |
| `utils.py`              | Collection of utilitiy functions.                            |
| `main.py`               | The main program to setup the training pipeline and perform training. |
| `inference.py`          | Deploy trained models to infer the segmetation areas. Also visualize some results. |
| `dice_score.py`         | Evalute the model performance by computing the DICE score. Also visualize some results. |
| `./data`                | This directory holds the data from the Duke dataset and UMN dataset, which we already converted from `.mat` files to `.png` images. Each dataset has four sub-directories: `./images` contains OCT images; `./labels` contains manual segmentation images from Doctor #1, which we used as the training/testing set; `./manualFuild2` contains images from Doctor #2 which we didn't touch too much. `./split` contains text files on how we set up the train/valid sets in cross-validations. |
| `./plots`               | This directory holds quantitative evaluation results for training/testing on the Duke dataset. There are two sub-directories, with `./DSC_Sensitivity_Specificity` holding all results we get while `./V2` holds results we used in the report and poster. |
| `./work_dirs`           | This directory mainly holds automatic segmentation results from our models. `./duke_dataset` contains results from models trained/tested on the Duke dataset. `./comparison_duke_and_umn` contains results from the Duke dataset and UMN dataset individually and provides a Python script to generate a `.svg` graph comparing the DICE scores from the two datasets. |
| `./tutorial`            | This directory contains the tutorial downloaded from MMEsegmentation official documentation website ([link](https://github.com/open-mmlab/mmsegmentation/blob/master/demo/MMSegmentation_Tutorial.ipynb)). |



## Project Overview

### Segmentation of OCT Images using a Transfer Learning Approach

We have utilized the MM-SEGMENTATION framework and PyTorch for training and evaluation of our models.

### Models Used

- **Swin Transformers**
- **Deeplabv3+**
- **Mobilenetv3**

## Getting Started

### Prerequisites

- Python
- Docker (recommended)

## Data Preparation

### Converting OCT Images from .mat to .png

Before training the models, it is necessary to convert the OCT images from their original `.mat` format to `.png`. This is done using the `convert_mat_to_png.py` script.

#### How to Run the Conversion Script

To convert the `.mat` files to `.png` format, run the following command:

```bash
python convert_mat_to_png.py

### Running the Project

To run the project, execute the following command:

```bash
python main.py
