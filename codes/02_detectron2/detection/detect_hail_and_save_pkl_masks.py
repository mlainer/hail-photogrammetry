#!/usr/bin/env python
# coding: utf-8

"""
Detect hail and save segmentation masks using Detectron2.

This script utilizes the Detectron2 library to detect hail in images and save the corresponding segmentation masks as pickle files.

Requirements:
- detectron2: A computer vision library for object detection.

Make sure to install Detectron2 before running this script.

Usage:
1. Specify the experiment folder containing the trained model in 'experiment_folder'.
2. Configure the model and load weights from the experiment folder.
3. Set the detection threshold and other model parameters.
4. Iterate through specified runs and process tile images, saving the segmentation masks as pickle files.

"""

from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import ColorMode
from detectron2.config import get_cfg

import os
import cv2
import numpy as np
import glob
from pathlib import Path
import pickle
import numpy as np

# Experiment folder containing the trained model
experiment_folder = './output/logs/hparam_tuning/run-3/'

# Load configuration
cfg = get_cfg()
cfg.merge_from_file("/home/appuser/detectron2_repo/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
cfg.MODEL.WEIGHTS = os.path.join(experiment_folder, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.90
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256
cfg.TEST.DETECTIONS_PER_IMAGE = 50

predictor = DefaultPredictor(cfg)

# Runs for analyzing the melting process
runs = ['hail_20220628_r1']

for run in runs:
    # Process tile images
    images_path = 'data/' + run + '/'

    # Output folder of pkl files
    mask_array_path = 'products/hparam_tuning/run-3/' + run + '/pkl/'

    if not os.path.exists(mask_array_path):
        os.makedirs(mask_array_path)

    all_images = glob.glob(images_path + '*.png')
    all_images.sort()

    for file in all_images:
        im = cv2.imread(file)  # Be sure that file has permissions 644
        image_name = Path(file).stem

        # Run the model for predictions
        outputs = predictor(im)
        masks = outputs['instances'][outputs['instances'].pred_classes == 0].pred_masks.cpu().numpy()

        mask_array = []
        for i in range(masks.shape[0]):
            mask_int = masks[i, :, :] * 1
            mask_array.append(mask_int)

        # Save the segmentation masks as pickle files
        with open(mask_array_path + 'mask_arrimage_name+'.pkl','wb') as f:
            pickle.dump(mask_array, f)
            