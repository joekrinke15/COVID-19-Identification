"""
copy the data files into train/val/test subdirectories

"""
import os
import re
from sklearn.model_selection import train_test_split
from glob import glob
import shutil
from tqdm import tqdm


DATA_DIR = "./data/super_set"
IMG_PATHS = glob(f'{DATA_DIR}/*/*.png')
split_regex = re.compile(r'[\\/]')
labels = []
for impath in tqdm(IMG_PATHS):
    *_, src, fname = re.split(split_regex, impath)
    if(src == 'NORMAL'):
        label = 0
        pass
    elif(src == 'OTHER'):
        label = 1
        pass
    else:
        label = 2
        pass
    labels.append(label)

train_paths, test_paths, y_train, y_test = train_test_split(IMG_PATHS, labels,
                                                            test_size=0.2,
                                                            stratify=labels,
                                                            random_state=42)

train_paths, val_paths, y_train, y_val = train_test_split(train_paths,
                                                          y_train,
                                                          test_size=0.125,
                                                          stratify=y_train,
                                                          random_state=42)

def move_subset(path_list, subset):
    for path in tqdm(path_list):
        *_, src, fname = re.split(split_regex, path)
        dst_dir = f'{DATA_DIR}/{subset}/{src}/'
        if not os.path.isdir(dst_dir):
            os.makedirs(dst_dir)
            pass
        shutil.copy(path, f'{dst_dir}/{fname}')

move_subset(train_paths, 'train')
move_subset(val_paths, 'val')
move_subset(test_paths, 'test')