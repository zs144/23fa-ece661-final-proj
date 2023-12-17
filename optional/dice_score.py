import numpy as np
from PIL import Image
import os


def dice_score(y_true, y_pred):
    intersection = np.sum(y_true * y_pred)
    return (2. * intersection) / (np.sum(y_true) + np.sum(y_pred))



if __name__ == '__main__':

    manual_1 = f'D:/projects/phd_assignments/ece661/project/data/labels/'
    manual_2 = f'D:/projects/phd_assignments/ece661/project/data/manualFluid2/'


    manual_1_files = os.listdir(manual_1)
    manual_2_files = os.listdir(manual_2)


    common = set(manual_1_files) & set(manual_2_files)
    dsc_all = 0
    for file in common:

        man_1 = Image.open(f'{manual_1}/{file}')
        man_2 = Image.open(f'{manual_2}/{file}')

        man_1_array = np.array(man_1).flatten()
        man_2_array = np.array(man_2).flatten()

        dsc = dice_score(man_1_array, man_2_array)

        dsc_all+=dsc

    print ('avg dsc', dsc_all/len(common))


    print (len(common))