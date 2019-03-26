from __future__ import print_function
import os
import yaml
import time
import random
import shutil
# import argparse
import numpy as np
import pandas as pd
# # # os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
# from sklearn.model_selection import KFold

from sklearn.metrics import f1_score

from libs.utils import LiverDataset
# # from libs.utils import get_augumentor
from libs.utils import accuracy
from libs.utils import AverageMeter
from libs.utils import adjust_learning_rate
# # from libs.utils import save_checkpoint
# # from libs.utils import balance_accuracy

from libs.set_models import *

config_dir = "./configs/train_resnet34.yml"

with open(config_dir) as fp:
    cfg = yaml.load(fp)

train_transforms = transforms.Compose([
        transforms.ToTensor(),
        ])

test_transforms = transforms.Compose([
        transforms.ToTensor(),
        ])

BASE_PATH = cfg['dataset']['data']
TRAIN_LABEL_PATH = os.path.join(BASE_PATH, "train_choose_label.csv")
val_LABEL_PATH = os.path.join(BASE_PATH, "val_choose_label.csv")
train_df = pd.read_csv(TRAIN_LABEL_PATH)
val_df = pd.read_csv(val_LABEL_PATH)
train_df['id'] = BASE_PATH + "/train_dataset/" + train_df['id']
val_df['id'] = BASE_PATH + "/train_dataset/" + val_df['id']
train_df['suffix'] = '.npy'
val_df['suffix'] = '.npy'
train_dataset = LiverDataset(train_df, '', train_transforms)
val_dataset = LiverDataset(val_df, '', test_transforms)

train_loader = torch.utils.data.DataLoader(
    train_dataset, 1, shuffle=True,
    num_workers=cfg['training']['workers'], pin_memory=True)
val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=1, shuffle=False,
    num_workers=cfg['training']['workers'], pin_memory=True)

x_tot = np.zeros(9)
x2_tot = np.zeros(9)
count = 0

for x, _ in iter(train_loader):
    count += 1
    # print(x.size())
    x = x.view(9, -1)
    print(count)
    if count % 100 == 0:
        print(x_tot)
    x_tot += x.mean(dim=1).numpy()
    x2_tot += (x**2).mean(dim=1).numpy()

print('start val!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
for x, _ in iter(val_loader):
    count += 1
    x = x.view(9, -1)
    print(count)
    if count % 100 == 0:
        print(x_tot)
    x_tot += x.mean(dim=1).numpy()
    x2_tot += (x**2).mean(dim=1).numpy()
    
channel_avr = x_tot / len(train_loader)
channel_std = np.sqrt(x2_tot/len(train_loader) - channel_avr**2)
print(count)
print(channel_avr, channel_std)