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
EPOCHS = 400

# File path settings
# BASE_DIR = Path('/kaggle/input/hubmap-hacking-the-human-vasculature')
BASE_DIR = Path('/home/viktor/Documents/kaggle/hubmap-2023/kaggle-data')

print(device)


# ### Directories

# In[ ]:


def mkdir_yolo_data(train_path, val_path):
    """
    make yolo data's directories
    
    parameters
    ----------
    train_path: str
        path for training data
    val_path: str
        path for validation data
    
    returns
    ----------
    train_image_path: str
        path for images of training data
    train_label_path: str
        path for labels of trainingdata
    val_image_path: str
        path for images of validation data
    val_label_path: str
        path for labels of validation data
    """
    train_image_path = Path(f'{train_path}/images')
    train_label_path = Path(f'{train_path}/labels')
    val_image_path = Path(f'{val_path}/images')
    val_label_path = Path(f'{val_path}/labels')
    
    train_image_path.mkdir(parents=True, exist_ok=True)
    train_label_path.mkdir(parents=True, exist_ok=True)
    val_image_path.mkdir(parents=True, exist_ok=True)
    val_label_path.mkdir(parents=True, exist_ok=True)
    
    return train_image_path, train_label_path, val_image_path, val_label_path


# In[ ]:


test_paths = glob(f'{BASE_DIR}/test/*')
polygons_path = f'{BASE_DIR}/polygons.jsonl'

yolo_train_path = 'datasets/train'
yolo_val_path = 'datasets/val'


# In[ ]:


# make directories
train_image_path, train_label_path, val_image_path, val_label_path = mkdir_yolo_data(yolo_train_path, yolo_val_path)
print(train_image_path)
print(train_label_path)
print(val_image_path)
print(val_label_path)


# ## Create annotation files and move tif to yolo' directory

# In[ ]:


def create_vessel_annotations(polygons_path):
    """
    Create annotations set which have blood_vessel label.
    
    parameters
    ----------
    polygons_path: str
        path of polygons.jsonl
    
    returns
    ----------
    annotations_dict: dict {key=id, value=coordinates}
        annotations dict with key id and value coordinates of blood_vessel
    """
    # load polygons data
    polygons = pd.read_json(polygons_path, orient='records', lines=True)
    
    # extract blood_vessel annotation
    annotations_dict = defaultdict(list)
    for idx, row in polygons.iterrows():
        id_ = row['id']
        annotations = row['annotations']
        for annotation in annotations:
            if annotation['type'] == 'blood_vessel':
                annotations_dict[id_].append(annotation['coordinates'])
    
    return annotations_dict

def create_label_file(id_, coordinates, path):
    """
    Create label txt file for yolo v8
    
    parameters
    ----------
    id_: str
        label id
    coordinates: list
        coordinates of blood_vessel
    path: str
        path for saving label txt file
    """
    label_txt = ''
    for coordinate in coordinates:
        label_txt += '0 '
        # Normalize
        coor_array = np.array(coordinate[0]).astype(float)
        coor_array /= float(IMAGE_SIZE)
        # transform to str
        coor_list = list(coor_array.reshape(-1).astype(str))
        coor_str = ' '.join(coor_list)
        # add string to label txt
        label_txt += f'{coor_str}\n'
    
    # Write labels to txt file
    with open(f'{path}/{id_}.txt', 'w') as f:
        f.write(label_txt)
        
def prepare_yolo_dataset(
        annotaions_dict, train_image_path, train_label_path, 
        val_image_path, val_label_path):
    """
    Prepare yolo dataset with images and labels
    
    parameters
    ----------
    annotations_dict: dict {key=id, value=coordinates}
        annotations dict with key id and value coordinates of blood_vessel
    train_image_path: str
        path for images of training data
    train_label_path: str
        path for labels of trainingdata
    val_image_path: str
        path for images of validation data
    val_label_path: str
        path for labels of validation data
    """
    ids = list(annotations_dict.keys())
    
    # train test split
    indices = [i for i in range(len(ids))]
    train_indices, val_indices = train_test_split(indices, test_size=0.2, random_state=1234)
    
    # Training data
    for index in tqdm(train_indices):
        id_ = ids[index]
        
        # create label txt file
        create_label_file(id_, annotations_dict[id_], train_label_path)
        # copy tif image file to yolo directory
        source_file = f'{BASE_DIR}/train/{id_}.tif'
        shutil.copy2(source_file, train_image_path)
    
    # Validation data
    for index in tqdm(val_indices):
        id_ = ids[index]
        
        # create label txt file
        create_label_file(id_, annotations_dict[id_], val_label_path)
        # copy tif image file to yolo directory
        source_file = f'{BASE_DIR}/train/{id_}.tif'
        shutil.copy2(source_file, val_image_path)
    


# In[ ]:


# Create annotations dict with key=id and value=coordinates
annotations_dict = create_vessel_annotations(polygons_path)


# In[ ]:


# Prepare dataset for yolo training
prepare_yolo_dataset(
    annotations_dict, train_image_path, train_label_path,
    val_image_path, val_label_path
)


# ## YOLO

# In[ ]:


# Edit yaml content
yaml_content = f'''
train: train/images
val: val/images

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
model = YOLO('yolov8m-seg.pt')


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
    iou=0.6
)


# In[ ]:


get_ipython().system('ls runs/segment/train')


# In[ ]:


show_image(filename='runs/segment/train/val_batch0_pred.jpg')


# In[ ]:


show_image(filename='runs/segment/train/val_batch0_labels.jpg')


# In[ ]:


show_image(filename='runs/segment/train/results.png')


# In[ ]:


show_image(filename='runs/segment/train/MaskP_curve.png')


# In[ ]:


trained_model = YOLO('runs/segment/train/weights/best.pt')
results = list(trained_model.predict(test_paths, save=True, conf=0.6))
result = results[0]


# In[ ]:


get_ipython().system('ls runs/segment/predict')


# In[ ]:


image = tiff.imread('runs/segment/predict/72e40acccadf.tif')
plt.imshow(image)


# In[ ]:




