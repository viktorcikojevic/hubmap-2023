import base64
import numpy as np
from pycocotools import _mask as coco_mask
import typing as t
import zlib
import cv2 
from ultralytics import YOLO
import tifffile as tiff
from tqdm import tqdm
import pandas as pd
import skimage

def encode_binary_mask(mask: np.ndarray) -> t.Text:
  """Converts a binary mask into OID challenge encoding ascii text."""

  # check input mask --
  if mask.dtype != np.bool:
    raise ValueError(
        "encode_binary_mask expects a binary mask, received dtype == %s" %
        mask.dtype)

  mask = np.squeeze(mask)
  if len(mask.shape) != 2:
    raise ValueError(
        "encode_binary_mask expects a 2d mask, received shape == %s" %
        mask.shape)

  # convert input mask to expected COCO API input --
  mask_to_encode = mask.reshape(mask.shape[0], mask.shape[1], 1)
  mask_to_encode = mask_to_encode.astype(np.uint8)
  mask_to_encode = np.asfortranarray(mask_to_encode)

  # RLE encode mask --
  encoded_mask = coco_mask.encode(mask_to_encode)[0]["counts"]

  # compress and base64 encoding --
  binary_str = zlib.compress(encoded_mask, zlib.Z_BEST_COMPRESSION)
  base64_str = base64.b64encode(binary_str)
  return base64_str

import os
import numpy as np
import torch
from PIL import Image

def load_tf2image(path_tif_test):
    array = tiff.imread(path_tif_test)
    img_example = Image.fromarray(array)
    img = np.array(img_example)
    return img

def clean_predict_class_blood_vessel(bboxes, l_conf_score, np_masks):
    cleaned_bboxes = []
    cleaned_conf_score = []
    cleaned_masks = []
    cls_blood_vessel_id = 0
    for idx in range(len(bboxes)):
        cls_pred = bboxes[idx][5]
        mask_check = np_masks[idx,:,:]
        area = mask_check.sum().item()
        if area < 10:
            continue
        if cls_pred == cls_blood_vessel_id:
            cleaned_bboxes.append(bboxes[idx])
            cleaned_conf_score.append(l_conf_score[idx])
            cleaned_masks.append(np_masks[idx])
    return np.array(cleaned_bboxes), np.array(cleaned_conf_score), np.array(cleaned_masks)

def predict_and_encoded(model, image, threshold = 0.001):
    image_input = image.copy()
    H, W, _ = np.shape(image_input)
    image_rs = cv2.resize(image_input, (512, 512))
    
    results = model.predict(image_rs,\
                            conf = threshold,\
                            iou = 0.6,
                            verbose=False)
    visualize = np.zeros((H, W), dtype = np.uint8)
    pred_string = ""
    
    for result in results:
        boxes = result.boxes  # Boxes object for bbox outputs
        masks = result.masks  # Masks object for segmentation masks outputs
        probs = result.probs  # Class probabilities for classification outputs
        bboxes = boxes.data.detach().cpu().numpy()
        if masks == None:
            return H, W, pred_string
        
        
        l_conf_score = result.boxes.conf.detach().cpu().numpy()
        np_masks = masks.data.detach().cpu().numpy()
        
        bboxes,\
            l_conf_score,\
            np_masks = clean_predict_class_blood_vessel(bboxes,\
                                                l_conf_score,\
                                                np_masks)
        num_masks = len(np_masks)
        for idx_mask in range(num_masks):
            mask = np_masks[idx_mask,:,:]
            mask = cv2.resize(mask, (W, H))
            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=1)
            mask = mask.astype(np.bool)
            encoded = encode_binary_mask(mask)
            score = l_conf_score[idx_mask]
            if idx_mask==0:
                pred_string += f"0 {score} {encoded.decode('utf-8')}"
            else:
                pred_string += f" 0 {score} {encoded.decode('utf-8')}"
        
    return H, W, pred_string

def get_submission_file(l_ids, l_heights, l_widths, l_prediction_strings):
    submission = pd.DataFrame()
    submission['id'] = l_ids
    submission['height'] = l_heights
    submission['width'] = l_widths
    submission['prediction_string'] = l_prediction_strings
    submission = submission.set_index('id')
    submission.to_csv("/kaggle/working/submission.csv")
    submission.head()

def get_prefixname(nfile):
    tmp = nfile.split(".")
    len_extension = len(tmp[-1]) + 1
    return nfile[:-len_extension]

import time
import os

def predict_test_dataset(path_test_dataset, model):
    l_ids = []
    l_heights = []
    l_widths = []
    l_prediction_strings = []
    for nfile in tqdm(os.listdir(path_test_dataset)):
        image = load_tf2image(path_test_dataset + "/" + nfile)
        H, W, pred_string = predict_and_encoded(model, image)
        id = get_prefixname(nfile)
        l_ids.append(id)
        l_heights.append(H)
        l_widths.append(W)
        l_prediction_strings.append(pred_string)
    return l_ids, l_heights, l_widths, l_prediction_strings

def refine_mask():
    path_test_dataset = "/kaggle/input/hubmap-hacking-the-human-vasculature/test"
    path_checkpoint_store = "/kaggle/input/fold0besthubmapcheckpoint0702/fold_0_best.pt"

    model = YOLO(path_checkpoint_store)

    l_ids, l_heights, l_widths, l_prediction_strings = predict_test_dataset(path_test_dataset, model)
    get_submission_file(l_ids, l_heights, l_widths, l_prediction_strings)