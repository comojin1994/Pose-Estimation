import numpy as np
import tensorflow as tf

MPII_PATH = f'../../../Datasets/MPII/'
IMG_PATH = f'{MPII_PATH}/images/'

TRAIN_CSV = f'{MPII_PATH}/annotation/train.csv'
TEST_CSV = f'{MPII_PATH}/annotation/test.csv'

CKPT_PATH = f'./checkpoints/'

size = 416
num_classes = 17
learning_rate = 1e-3
EPOCHS = 2

yolo_anchors = np.array([(10, 13), (16, 30), (33, 23), (30, 61), (62, 45),
                         (59, 119), (116, 90), (156, 198), (373, 326)],
                        np.float32) / 416
yolo_anchor_masks = np.array([[6, 7, 8], [3, 4, 5], [0, 1, 2]])