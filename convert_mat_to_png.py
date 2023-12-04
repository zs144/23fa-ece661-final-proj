#Load Required Libraries

import numpy as np
from scipy.io import loadmat
from pathlib import Path
import os
from PIL import Image


def check_for_nonzero(item):
    """
    Check for Non-zero in numpy array
    """
    if np.count_nonzero(item)!=0:
        return True
    return False


def nan_to_num(data):
    """
    Convert nan values in numpy array to real number
    """
    image_data = np.nan_to_num(data)
    return image_data

def main(fpath, save_path):

    """
    Convert the data in mat format to PNG images for the subjects where segmentation exists
    """

    save_path_image = f'{save_path}/images/'
    save_path_label = f'{save_path}/multi_label/'

    if not os.path.exists(save_path_image):
        os.makedirs(save_path_image)

    if not os.path.exists(save_path_label):
        os.makedirs(save_path_label)




    subject_id = ['01', '02', '03', '04','05', '06','07', '08','09', '10']

    for id in subject_id:

        example_name = f'Subject_{id}.mat'
        mat_data = loadmat(f"{fpath}/{example_name}")

        for index in range(mat_data["images"].shape[2]):

            label_slice = mat_data["manualFluid1"][:, :, index]
            label_slice = nan_to_num(label_slice)

            if check_for_nonzero(label_slice):

                image_slice = nan_to_num(mat_data["images"][:, :, index])
                image = Image.fromarray(image_slice.astype(np.uint8))
                if image.mode != 'RGB':
                    image = image.convert('RGB')

                image.save(f"{save_path_image}/Subject_{id}_{index}.png")

                #Convert the label image to binary label
                # label_slice[label_slice != 0] = 1
                label = Image.fromarray(label_slice.astype(np.uint8))
                label.save(f"{save_path_label}/Subject_{id}_{index}.png")


if __name__ =='__main__':


    load_path  = Path(f'D:/projects/phd_assignments/ece661/project/archive/2015_BOE_Chiu/2015_BOE_Chiu/')
    save_path = Path(f'D:/projects/phd_assignments/ece661/project/data/')


    main(load_path, save_path)
