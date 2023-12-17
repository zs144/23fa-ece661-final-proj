# ECE 661 Final Project Repository

This repository contains the source code and related materials for the ECE 661 final project by Lavsen, Ken, and Zion.

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
