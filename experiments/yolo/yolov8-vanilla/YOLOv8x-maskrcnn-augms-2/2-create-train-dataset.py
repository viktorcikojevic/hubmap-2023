#!/usr/bin/env python
# coding: utf-8

# In[16]:


import os, glob
import sys
import json
from PIL import Image
from collections import Counter

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import tifffile as tiff
import matplotlib.pyplot as plt
from tqdm import tqdm

import cv2

from sklearn.model_selection import KFold

sys.path.append("detection-wheel")


# In[17]:


EPOCHS = 60


# In[18]:


import transforms as T


import albumentations as A
from albumentations.pytorch import ToTensorV2
import os
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as F, InterpolationMode
from skimage.draw import polygon

class PennFudanDataset(torch.utils.data.Dataset):
    def __init__(self, imgs, masks, mode='train'):
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = imgs
        self.masks = masks

        self.mode = mode
        
        
        labels_file = "/home/viktor/Documents/kaggle/hubmap-2023/kaggle-data/polygons.jsonl"
        with open(labels_file, 'r') as json_file:
            self.json_labels = [json.loads(line) for line in json_file]
            
        # get index for each 'id' 
        indices_map = {}
        for indx, label in enumerate(self.json_labels):
            indices_map[label['id']] = indx
        
        self.indices_map = indices_map
        
        if self.mode == 'train':
            self.alb_transform = A.Compose([
                
                
                    A.Rotate(limit=90, p=0.9),
                    A.ShiftScaleRotate(p=0.5),
                    A.RandomBrightnessContrast(p=0.2),
                    A.RGBShift(r_shift_limit=30, g_shift_limit=30, b_shift_limit=30, p=0.2),
                    A.RandomGamma(p=0.2),
                    # A.RandomCrop(height=256, width=256, p=0.5),
                    A.RandomResizedCrop(height=512, width=512, p=0.5),
                    A.Affine(p=0.5),
                    A.Downscale(scale_min=0.1, scale_max=0.5, p=0.5),
                    A.CoarseDropout(max_holes=8, max_height=32, max_width=32, mask_fill_value=0, p=0.3),
                    A.ShiftScaleRotate(shift_limit=0, scale_limit=0, p=0.3),
                    
                    
                    A.Normalize(
                        mean= [0] * 3,
                        std= [1] * 3
                    ),
                
                
                    ToTensorV2(transpose_mask=True),
                ])
            
            self.random_zoom_out = T.Compose([T.RandomZoomOut(p=0.8)])
        else:
            self.alb_transform = A.Compose([
                    
                    A.Normalize(
                        mean= [0] * 3,
                        std= [1] * 3
                    ),
                
                
                    ToTensorV2(transpose_mask=True),
                ])
            
        

    def __getitem__(self, idx):
        
        num_objs = 0
        while num_objs == 0:
        
            idx = np.random.randint(0, len(self.imgs))
            
            # load images and masks
            img_path = self.imgs[idx]
            mask_path = self.masks[idx] # '/home/viktor/Documents/kaggle/hubmap-2023/experiments/mask-rcnn/new-dataset/train/mask/0006ff2aa7cd_mask.png'
            
            
            
            # load image
            img = Image.open(img_path).convert("RGB")
            img = np.array(img)
            
            # load mask, but first get the id
            mask_id = mask_path.split('/')[-1].split('_')[0]
            
            mask = np.zeros((512, 512), dtype=np.float32)
            
            
            mask_id_indx = self.indices_map[mask_id]
            for annot in self.json_labels[mask_id_indx]['annotations']:
                cords = annot['coordinates']
                if annot['type'] == "blood_vessel":
                    for cord in cords:
                        rr, cc = polygon(np.array([i[1] for i in cord]), np.asarray([i[0] for i in cord]))
                        mask[rr, cc] = 1
            
        
            # ---------------- Augmentations ----------------
            
            
            
            # albumentations
            transformed = self.alb_transform(image=img, mask=mask)
            
            img = transformed['image']
            mask = transformed['mask'].numpy()
            
            # return img, mask
            
            # print(img.shape, mask.shape)
            mask_uint8 = np.where(mask > 0.5, 1, 0).astype(np.uint8) * 255
                
            num_outputs, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_uint8)
            label_masks = [labels == i for i in range(num_outputs)]
            masks = []
            for m in label_masks:
                mask_m = mask * m
                if np.sum(mask_m) > 10:
                    masks.append(mask_m)
                    
            
            # -----------------------------------------------
            
            # get bounding box coordinates for each mask
            num_objs = len(masks)
            boxes = []
            
            
            
            
            
        
        
        for i in range(num_objs):
            pos = np.nonzero(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])
    
    
        

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        image_id = torch.tensor([idx])
        
        try:
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
            #print(area,area.shape,area.dtype)
        except:
            area = torch.tensor([[0],[0]])
        
        
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
        
        
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target['num_objs'] = torch.tensor(num_objs)      
        target["image_id"] = image_id  
        target["area"] = area
        target["iscrowd"] = iscrowd

                
        if self.mode == 'train':
            
            rand_width_height = 512 # np.random.randint(512, 1024)
            # resize to 1024x1024
            new_width = rand_width_height
            new_height = rand_width_height
            orig_height, orig_width = img.shape[1:]
            # orig_height = img.size[1]
            # orig_width = img.size[0]
            
            img = F.resize(img, [new_height, new_width], interpolation=InterpolationMode.BILINEAR)

            if target is not None:
                target["boxes"][:, 0::2] *= new_width / orig_width
                target["boxes"][:, 1::2] *= new_height / orig_height
                if "masks" in target:
                    target["masks"] = F.resize(
                        target["masks"], [new_height, new_width], interpolation=InterpolationMode.NEAREST
                    )
            
        
        return img, target

    def __len__(self):
        return len(self.imgs)


# In[19]:


import transforms as T

def get_transform(train):
    transforms = []
    transforms.append(T.PILToTensor())
    transforms.append(T.ConvertImageDtype(torch.float))
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
        transforms.append(T.RandomIoUCrop())
        transforms.append(T.RandomZoomOut())
        transforms.append(T.RandomPhotometricDistort())
        transforms.append(T.ScaleJitter())
        transforms.append(T.RandomShortestSize())
        # transforms.append(T.FixedSizeCrop(size=(512,512)))
        
        
    return T.Compose(transforms)


# In[20]:


from engine import train_one_epoch, evaluate
import utils


# In[21]:


n_imgs = len(glob.glob('/home/viktor/Documents/kaggle/hubmap-2023/experiments/mask-rcnn/new-dataset/train/image/*'))
n_imgs



# In[22]:


from sklearn.model_selection import train_test_split

all_imgs = sorted(glob.glob('/home/viktor/Documents/kaggle/hubmap-2023/experiments/mask-rcnn/new-dataset/train/image/*.png'))
all_masks = sorted(glob.glob('/home/viktor/Documents/kaggle/hubmap-2023/experiments/mask-rcnn/new-dataset/train/mask/*.png'))
all_imgs = np.array(all_imgs)
all_masks = np.array(all_masks)


indices = [i for i in range(len(all_imgs))]
train_indices, val_indices = train_test_split(indices, test_size=0.2, random_state=1234)

train_img = all_imgs[train_indices]
train_mask = all_masks[train_indices]
val_img = all_imgs[val_indices]
val_mask = all_masks[val_indices]
dataset_train = PennFudanDataset(train_img, train_mask, mode='train')
# dataset_val = PennFudanDataset(val_img, val_mask, get_transform(train=False), mode='val')
train_dl = torch.utils.data.DataLoader(
    dataset_train, batch_size=1, shuffle=True, num_workers=os.cpu_count(), pin_memory=True, drop_last=True, collate_fn=utils.collate_fn)
# val_dl = torch.utils.data.DataLoader(dataset_val, batch_size=1, shuffle=False, num_workers=os.cpu_count(), pin_memory=True,collate_fn=utils.collate_fn)


# In[23]:


img, target = train_dl.dataset[0]
# img.shape, target['masks']
target['masks'].shape, img.shape, img.min(), img.max() # -> (torch.Size([8, 512, 512]), torch.Size([3, 512, 512]), tensor(0.), tensor(1.))


# # Create train dataset which is an augmentation of the original train dataset

# In[24]:


# # load /home/viktor/Documents/kaggle/hubmap-2023/experiments/yolo/yolov8-vanilla/YOLOv8x-maskrcnn-augms/datasets/train/images/0a4ddecc55f0.tif
# img = tiff.imread('/home/viktor/Documents/kaggle/hubmap-2023/experiments/yolo/yolov8-vanilla/YOLOv8x-maskrcnn-augms/datasets/train/images/0a4ddecc55f0.tif')
# img.min(), img.max(), img.shape # -> (0, 255, (512, 512, 3))


# In[25]:


mask = target['masks'][0]
mask.shape # -> torch.Size([512, 512])


# In[26]:


import cv2
import numpy as np

# Assuming mask is a 2D torch tensor of shape [H, W]
mask_np = mask.cpu().numpy() # Convert the mask to a numpy array

# Find contours. Note, this will give the boundary pixels, 
# so polygons will be represented as a list of points along the boundary
contours, _ = cv2.findContours(mask_np.astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

polygons = [cnt.reshape(-1, 2) for cnt in contours] # Reshape for easier handling

# Each element in polygons now represent a separate connected component in the original mask
# polygons # [array([[121,  10], [120,  11], [119,  11],


# In[27]:


folder_root = "datasets/train-2/"


# In[28]:


img.shape


# In[29]:


import torch
from PIL import Image

# Assuming that torch_tensor is your image tensor
torch_tensor = img

# Denormalize the tensor from [0,1] to [0,255]
torch_tensor_denorm = (torch_tensor * 255).byte()

# Permute the tensor to make it suitable for creating a PIL image
img_pil = Image.fromarray(torch_tensor_denorm.cpu().numpy().transpose(1,2,0))

# Save the image
img_pil.save('image.tif')


# In[30]:


from tqdm import tqdm

n_epochs = 4

train_dl_indx = 0

for epoch in range(n_epochs):
    for images, targets in tqdm(train_dl):
        
        img = images[0]
        target = targets[0]
        masks = target['masks']
        
        
        # save img 
        # Assuming that torch_tensor is your image tensor
        torch_tensor = img

        # Denormalize the tensor from [0,1] to [0,255]
        torch_tensor_denorm = (torch_tensor * 255).byte()

        # Permute the tensor to make it suitable for creating a PIL image
        img_pil = Image.fromarray(torch_tensor_denorm.cpu().numpy().transpose(1,2,0))

        # Save the image
        img_pil.save(f'datasets/train/images/{train_dl_indx}_2.tif')
        

        # save the mask
        
        label_txt = ''
        
        
        for mask in masks:
        
        
            # get polygons
            # Assuming mask is a 2D torch tensor of shape [H, W]
            mask_np = mask.cpu().numpy() # Convert the mask to a numpy array

            # Find contours. Note, this will give the boundary pixels, 
            # so polygons will be represented as a list of points along the boundary
            contours, _ = cv2.findContours(mask_np.astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            coordinates = [cnt.reshape(-1, 2) for cnt in contours] # Reshape for easier handling
            
            
            label_txt += '0 '
            # Normalize
            coor_array = np.array(coordinates[0]).astype(float)
            coor_array /= float(512)
            # transform to str
            coor_list = list(coor_array.reshape(-1).astype(str))
            coor_str = ' '.join(coor_list)
            # add string to label txt
            label_txt += f'{coor_str}\n'
        
        # delete f'datasets/train/labels/{train_dl_indx}.txt'
        if os.path.exists(f'datasets/train/labels/{train_dl_indx}_2.txt'):
            os.remove(f'datasets/train/labels/{train_dl_indx}_2.txt')
            
        # Write labels to txt file
        
        with open(f'datasets/train/labels/{train_dl_indx}_2.txt', 'w') as f:
            f.write(label_txt)
        
        train_dl_indx += 1

