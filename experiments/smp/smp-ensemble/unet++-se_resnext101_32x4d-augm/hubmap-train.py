#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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


# In[ ]:


import torch


# In[ ]:


class CFG:
    LR = 3e-5
    EPOCHS = 800
    BATCH_SIZE = 5
    N_TRAIN = 1400 # Take first N_TRAIN images for training, rest for validation


    encoder_name = "se_resnext101_32x4d"
    encoder_depth = 4
    decoder_channels = [512, 256, 128, 64]
     
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# In[ ]:


import os
import numpy as np
import torch
from PIL import Image
# import torch functional as F
import torch.nn.functional as F

class PennFudanDataset(torch.utils.data.Dataset):
    def __init__(self, imgs, masks, transforms):
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = imgs#sorted(glob.glob('/home/viktor/Documents/kaggle/hubmap-2023/experiments/mask-rcnn/new-dataset/train/image/*.png'))
        self.masks = masks#sorted(glob.glob('/home/viktor/Documents/kaggle/hubmap-2023/experiments/mask-rcnn/new-dataset/train/mask/*.png'))

    def __getitem__(self, idx):
        # load images and masks
        img_path = self.imgs[idx]
        mask_path = self.masks[idx]
        img = Image.open(img_path).convert("RGB")
        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background
        mask = Image.open(mask_path).convert('L')
        # convert the PIL Image into a numpy array
        mask = np.array(mask)
        # instances are encoded as different colors
        obj_ids = np.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]

        # split the color-encoded mask into a set
        # of binary masks
        #masks = (mask == obj_ids[:, None, None])
        #print((obj_ids[:, None, None]).shape)
        #masks = mask == obj_ids[:, None, None]
        masks = [np.where(mask== obj_ids[i, None, None],1,0) for i in range(len(obj_ids))]
        masks = np.array(masks)

        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
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
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
        
        #print(masks.shape)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)


        target = target["masks"]
        target = torch.sum(target, dim=0)
        
        
        img = F.interpolate(img.unsqueeze(0), size=(512, 512), mode='bilinear', align_corners=False).squeeze()
        target = F.interpolate(target.unsqueeze(0).unsqueeze(0).float(), size=(512, 512), mode='bilinear', align_corners=False).squeeze()
        
        
        
        return img, target

    def __len__(self):
        return len(self.imgs)


# In[ ]:


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
        # transforms.append(T.RandomShortestSize())
    return T.Compose(transforms)


# In[ ]:


import segmentation_models_pytorch as smp


# In[ ]:


import segmentation_models_pytorch as smp
DiceLoss = smp.losses.DiceLoss(mode='binary')
BCELoss = smp.losses.SoftBCEWithLogitsLoss(pos_weight=torch.tensor(1.0).to(CFG.device))
FocalLoss = smp.losses.FocalLoss(mode='binary')

def criterion(y_pred, y_true):
    
    
    # return BCELoss(y_pred, y_true)
    return 0.5 * BCELoss(y_pred, y_true) + 0.5 * DiceLoss(y_pred, y_true)


# In[ ]:


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


# In[ ]:


n_imgs = len(glob.glob('/home/viktor/Documents/kaggle/hubmap-2023/experiments/mask-rcnn/new-dataset/train/image/*'))
n_imgs



# In[ ]:


# import sys

# # Create a custom function to log output
# def log_output(text):
#     with open('output.log', 'a') as f:
#         f.write(text)

# # Redirect stdout to the custom log function
# sys.stdout.write = log_output


# In[ ]:


import utils


# In[ ]:


kf = KFold(n_splits=5, shuffle=True, random_state=43)
for i, (train_index, test_index) in enumerate(kf.split(range(n_imgs))):
    if i!=0: continue
    all_imgs = sorted(glob.glob('/home/viktor/Documents/kaggle/hubmap-2023/experiments/mask-rcnn/new-dataset/train/image/*.png'))
    all_masks = sorted(glob.glob('/home/viktor/Documents/kaggle/hubmap-2023/experiments/mask-rcnn/new-dataset/train/mask/*.png'))
    all_imgs = np.array(all_imgs)
    all_masks = np.array(all_masks)
    train_img = all_imgs[train_index]
    train_mask = all_masks[train_index]
    val_img = all_imgs[test_index]
    val_mask = all_masks[test_index]
    dataset_train = PennFudanDataset(train_img, train_mask, get_transform(train=True))
    dataset_val = PennFudanDataset(val_img, val_mask, get_transform(train=False))
    train_dl = torch.utils.data.DataLoader(
        dataset_train, batch_size=4, shuffle=True, num_workers=os.cpu_count(), pin_memory=True, drop_last=True)
    val_dl = torch.utils.data.DataLoader(
        dataset_val, batch_size=1, shuffle=False, num_workers=os.cpu_count(), pin_memory=True)
    
    model = smp.UnetPlusPlus(encoder_name=CFG.encoder_name, activation=None, encoder_depth=CFG.encoder_depth, decoder_channels=CFG.decoder_channels,encoder_weights=None)
    model = model.to(CFG.device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=3e-5)
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    # set linear warmup scheduler, with constant learning rate after warmup
    # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.0001,
    #                                             steps_per_epoch=10, epochs=EPOCHS//10,
    #                                             pct_start=0.01)

    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer,start_factor=0.01,
                                            end_factor=1,
                                            total_iters=10)

    
    for epoch in range(CFG.EPOCHS):
        
        ## -------------------- TRAIN --------------------
        model.train()
        
        
        pbar_train = enumerate(train_dl)
        pbar_train = tqdm(pbar_train, total=len(train_dl), bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}")
        
        mloss_train = 0.

        for _, (images, masks) in pbar_train:
            
            optimizer.zero_grad()   
            
            
            images, masks = images.to(CFG.device), masks.to(CFG.device)
            masks = masks.float()
            
            pred_mask = model(images)
            loss = criterion(pred_mask.squeeze(), masks.squeeze())
            
            
            loss.backward()    
            mloss_train += loss.detach().item()
            optimizer.step()
            
            
            # optimizer.zero_grad()   
            
            # for image, mask in zip(images, masks):
                
            #     image, mask = image.to(CFG.device), mask.to(CFG.device)
            #     image = image.unsqueeze(0)
            #     mask = mask.unsqueeze(0).float()
                
                
            #     # resize image to 512x512
            #     image = F.interpolate(image, size=(512, 512), mode='bilinear', align_corners=False)
            #     mask = F.interpolate(mask.unsqueeze(0), size=(512, 512), mode='bilinear', align_corners=False).squeeze(0)
                
                
                
                
            #     pred_mask = model(image)
            #     loss = criterion(pred_mask.squeeze(), mask.squeeze())
                
            #     total_loss += loss
            
            
            # total_loss.backward()    
            # mloss_train += total_loss.detach().item()
            # optimizer.step()
            
            # mloss_train += total_loss.detach().item()
        
        
        out = {}
        out['epoch'] = epoch
        out['loss_train'] = mloss_train / len(train_dl)
        
        
        ## -------------------- VALID --------------------
        
        model.eval()
        
        pbar_train = enumerate(val_dl)
        pbar_train = tqdm(pbar_train, total=len(val_dl), bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}")
        
        mloss_train = 0.

        for _, (images, masks) in pbar_train:
            
            
            images, masks = images.to(CFG.device), masks.to(CFG.device)
            
            with torch.no_grad():
                pred_mask = model(images)
                loss = criterion(pred_mask.squeeze(), masks.squeeze())
            
            
            mloss_train += loss.detach().item()
        
            
            # optimizer.zero_grad()
            
            # for image, mask in zip(images, masks):
                
            #     image, mask = image.to(CFG.device), mask.to(CFG.device)
            #     image = image.unsqueeze(0)
            #     mask = mask.unsqueeze(0).float()
                
                
            #     # resize image to 512x512
            #     image = F.interpolate(image, size=(512, 512), mode='bilinear', align_corners=False)
            #     mask = F.interpolate(mask.unsqueeze(0), size=(512, 512), mode='bilinear', align_corners=False).squeeze(0)
                
                
                
            #     with torch.no_grad():
            #         pred_mask = model(images)
            #         loss = criterion(pred_mask.squeeze(), mask.squeeze())
            
            #         # loss.backward()
            #         mloss_train += loss.detach().item()
            # # optimizer.step()
            
            # mloss_train += loss.detach().item()
        
        
        out['epoch'] = epoch
        out['loss_val'] = mloss_train / len(val_dl)
        
        ## -------------------- PRINT --------------------
        
        # get learning rate value
        lr = optimizer.param_groups[0]["lr"]
        out["lr"] = lr
        
        if epoch == 0:
            df_out = pd.DataFrame(out, index=[0])
            df_out.to_csv('log.csv', index=False)
        else:
            df_out = pd.read_csv('log.csv')
            df_out = pd.concat([df_out, pd.DataFrame(out, index=[0])])
            df_out.to_csv('log.csv', index=False)
        
        
        scheduler.step()
        
        if epoch % 10 == 0:
            model_path = f'ckpts/fold_{i}_epoch{epoch}.pth'
            torch.save(model.state_dict(), model_path)
        


# In[ ]:




