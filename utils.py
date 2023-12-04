import subprocess
import mmengine
import os.path as osp



def download_model(config_name, destination_folder):
    try:
        subprocess.run(['mim', 'download', 'mmsegmentation', '--config', config_name, '--dest', destination_folder], check=True)
        print(f"Model {config_name} downloaded successfully to {destination_folder}")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e}")

def create_train_val_split(data_root, ann_dir):

    """
    Modify this so that the all images from same patients are in same split to not have data leakage.
    """
    # split train/val set randomly
    split_dir = 'splits'
    mmengine.mkdir_or_exist(osp.join(data_root, split_dir))
    filename_list = [osp.splitext(filename)[0] for filename in mmengine.scandir(
        osp.join(data_root, ann_dir), suffix='.png')]
    with open(osp.join(data_root, split_dir, 'train.txt'), 'w') as f:
    # select first 4/5 as train set
        train_length = int(len(filename_list)*4/5)
        f.writelines(line + '\n' for line in filename_list[:train_length])
        with open(osp.join(data_root, split_dir, 'val.txt'), 'w') as f:
            # select last 1/5 as train set
            f.writelines(line + '\n' for line in filename_list[train_length:])