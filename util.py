import os
from pathlib import Path
import shutil

import numpy as np
from PIL import Image
import tifffile as tiff

#IMAGES_PATH = "images"
#RAW_PATH = "raw"
LR_10M_PATH = "data_10m"
GT_250CM_PATH = "data_250cm"
TEST_PATH = "test"
TEST_LABELS_PATH = "test_labels"
TRAIN_PATH = "train"
TRAIN_LABELS_PATH = "train_labels"
ROWS, COLS, CHANNELS = (953, 902, 3)
BIT_DEPTH = 8
MAX_VAL = 2 ** 8 - 1


def clean_mkdir(path):
    if Path(path).exists():
        shutil.rmtree(path)
        #os.rmdir(path)

    os.makedirs(path)



def load_run_data(run_path):
    x = []
    index = 0
    for filename in os.listdir(run_path):
        index += 1
        img = Image.open(run_path +"/"+ filename)
        img_array = np.asarray(img, dtype="uint8")
        img_array = img_array / (MAX_VAL * 1.0)
        x.append(img_array)
    return np.array(x)