#!/usr/bin/env python
# coding: utf-8

# ### YOLO v8 train & inference
# 
# We use the YOLO V8 model for this competition because it can execute the object detection and segmentation at the same time.  
# Because of this notebook is online, we can't submit this directly.  

# In[ ]:


import shutil
import os
import pandas as pd
import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt

from pathlib import Path
from glob import glob
from collections import defaultdict
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from IPython.display import Image as show_image

import ultralytics
from ultralytics import YOLO

import torch

ultralytics.checks()


# ## Set parameters

# ### Hyper parameters

# In[ ]:


device = 'cuda' if torch.cuda.is_available() else 'cpu'

IMAGE_SIZE = 512
BATCH_SIZE = 16
EPOCHS = 100


print(device)


# ## YOLO

# In[ ]:


# Edit yaml content
yaml_content = f'''
train: /home/viktor/Documents/kaggle/hubmap-2023/experiments/yolo/yolov8-vanilla/YOLOv8x-maskrcnn-augms-2/datasets/train/images
val: /home/viktor/Documents/kaggle/hubmap-2023/experiments/yolo/yolov8-vanilla/YOLOv8x-maskrcnn-augms-2/datasets/val/images

names:
    0: blood_vessel
'''

yaml_file = 'data.yaml'

with open(yaml_file, 'w') as f:
    f.write(yaml_content)


# ![image.png](attachment:image.png)

# In[ ]:


# prepare model
# model = YOLO('yolov8n-seg.pt')
model = YOLO('yolov8x-seg.pt')


# In[ ]:


x = torch.randn((1, 3, IMAGE_SIZE, IMAGE_SIZE))
pred = model.predict(x)


# In[ ]:


# training
results = model.train(
    batch=BATCH_SIZE,
    device=0,
    data=yaml_file,
    epochs=EPOCHS,
    imgsz=IMAGE_SIZE,
    mixup=0.2,
    flipud=0.2,
    shear=0.4,
    degrees=45,
    copy_paste=0.2,
    dropout=0.2,
    iou=0.6,
    lr0=0.0001,
    lrf=0.01,
    optimizer='Adam',
    max_det=100
)


# In[ ]:


image = tiff.imread('runs/segment/predict/72e40acccadf.tif')
plt.imshow(image)


# In[ ]:


# how many files in folder /home/viktor/Documents/kaggle/hubmap-2023/experiments/yolo/yolov8-vanilla/YOLOv8x-maskrcnn-augms/datasets/train/images
folder = '/home/viktor/Documents/kaggle/hubmap-2023/experiments/yolo/yolov8-vanilla/YOLOv8x-maskrcnn-augms/datasets/train/images'
len(os.listdir(folder))

