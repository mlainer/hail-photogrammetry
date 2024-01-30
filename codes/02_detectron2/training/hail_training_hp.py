#!/usr/bin/env python
# coding: utf-8
"""
This script is designed for hyperparameter tuning of a Hail detection model using Detectron2.
It registers train, validation, and test datasets, performs hyperparameter search, and logs the
results with TensorBoard. The main functionality includes defining hyperparameters, configuring
TensorBoard, training the model, and evaluating it using COCO metrics during training.

The script utilizes the Detectron2 library for object detection, TensorBoard for logging, and
TensorFlow for hyperparameter tuning. It aims to provide a flexible and modular approach for
experimenting with different hyperparameter configurations.

Please ensure that the required packages and the Detectron2 library are installed before running
the script. You can customize the paths and configurations in the script to adapt it to your specific
dataset and experimental setup.
"""
import os
from pathlib import Path
import random
import logging
import datetime
import time
import numpy as np
import tensorflow as tf
import torch
from detectron2.data.datasets import register_coco_instances
from detectron2.data.catalog import DatasetCatalog, MetadataCatalog
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.config import get_cfg
from detectron2.utils.logger import log_every_n_seconds
from detectron2.data import DatasetMapper, build_detection_test_loader
from detectron2.engine.hooks import HookBase
from detectron2.utils.comm import comm
from detectron2.utils.visualizer import Visualizer
from tensorboard.plugins.hparams import api as hp
import cv2
import matplotlib.pyplot as plt

cfg = get_cfg()

# Constants for hyperparameter tuning metrics
AP_BBOX = 'AP_bbox'
AP50_BBOX = 'AP50_bbox'
AP75_BBOX = 'AP75_bbox'
AP_SEGM = 'AP_segm'
AP50_SEGM = 'AP50_segm'
AP75_SEGM = 'AP75_segm'

def register_datasets():
    """
    Register train, validation, and test datasets using COCO format.
    """
    # Register train dataset
    register_coco_instances("train_hail", {}, "./data/hail_20210620_r1/train/annotations/instances_default.json", "./data/hail_20210620_r1/train/images")
    dataset_dicts_train = DatasetCatalog.get("train_hail")
    hail_metadata_train = MetadataCatalog.get("train_hail")

    # Register validation dataset
    register_coco_instances("val_hail", {}, "./data/hail_20210620_r1/val/annotations/instances_default.json", "./data/hail_20210620_r1/val/images")
    dataset_dicts_val = DatasetCatalog.get("val_hail")
    hail_metadata_val = MetadataCatalog.get("val_hail")

    # Register test dataset
    register_coco_instances("test_hail", {}, "./data/hail_20210620_r1/test/annotations/instances_default.json", "./data/hail_20210620_r1/test/images")
    dataset_dicts_test = DatasetCatalog.get("test_hail")
    hail_metadata_test = MetadataCatalog.get("test_hail")

def setup_hyperparameters():
    """
    Set up hyperparameters for tuning.
    """
    HP_BASE_LR = hp.HParam('base_lr', hp.Discrete([0.0001, 0.00025, 0.0005, 0.001]))
    HP_GAMMA = hp.HParam('gamma', hp.Discrete([0.1, 0.5]))
    HP_BATCH_SIZE_PER_IMAGE = hp.HParam('batch_size_per_image', hp.Discrete([128, 256]))

    return HP_BASE_LR, HP_GAMMA, HP_BATCH_SIZE_PER_IMAGE

def configure_tensorboard():
    """
    Configure TensorBoard for logging hyperparameter tuning metrics.
    """
    with tf.summary.create_file_writer('output/logs/hparam_tuning').as_default():
        hp.hparams_config(
            hparams=[HP_BASE_LR, HP_GAMMA, HP_BATCH_SIZE_PER_IMAGE],
            metrics=[
                hp.Metric(AP_BBOX, display_name='AP_bbox'),
                hp.Metric(AP50_BBOX, display_name='AP50_bbox'),
                hp.Metric(AP75_BBOX, display_name='AP75_bbox'),
                hp.Metric(AP_SEGM, display_name='AP_segm'),
                hp.Metric(AP50_SEGM, display_name='AP50_segm'),
                hp.Metric(AP75_SEGM, display_name='AP75_segm')
            ],
        )

def main():
    register = 1
    if register == 1:
        register_datasets()

    HP_BASE_LR, HP_GAMMA, HP_BATCH_SIZE_PER_IMAGE = setup_hyperparameters()
    configure_tensorboard()

    session_num = 0
    for base_lr in HP_BASE_LR.domain.values:
        for gamma in HP_GAMMA.domain.values:
            for batch_size_per_image in HP_BATCH_SIZE_PER_IMAGE.domain.values:
                hparams = {
                    HP_BASE_LR: base_lr,
                    HP_GAMMA: gamma,
                    HP_BATCH_SIZE_PER_IMAGE: batch_size_per_image,
                }
                run_name = "run-%d" % session_num
                print('--- Starting trial: %s' % run_name)
                print({h.name: hparams[h] for h in hparams})
                run('output/logs/hparam_tuning/' + run_name, hparams, base_lr, gamma, batch_size_per_image)
                session_num += 1
                
    return hparams            

def train_hail_model(run_dir, base_lr, gamma, batch_size):
    """
    Train the Hail detection model using the specified hyperparameters.

    Args:
    - run_dir (str): Directory to save the training output.
    - base_lr (float): Learning rate.
    - gamma (float): Learning rate decay factor.
    - batch_size (int): Batch size per image.

    Returns:
    - tuple: Metrics (AP_bbox, AP50_bbox, AP75_bbox, AP_segm, AP50_segm, AP75_segm).
    """
    cfg.merge_from_file("/home/appuser/detectron2_repo/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    cfg.INPUT.MIN_SIZE_TRAIN = (500,)
    cfg.OUTPUT_DIR = run_dir
    cfg.DATASETS.TRAIN = ("train_hail",)
    cfg.DATASETS.TEST = ("val_hail",)
    cfg.DATALOADER.NUM_WORKERS = 1

    #cfg.INPUT.RANDOM_FLIP = "horizontal"
    #cfg.SOLVER.CHECKPOINT_PERIOD = 2000

    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = base_lr
    cfg.SOLVER.WARMUP_ITERS = 300
    cfg.SOLVER.MAX_ITER = 3000 #adjust up if val AP is still rising, adjust down if overfit
    cfg.SOLVER.STEPS = (2400, 2700)
    cfg.SOLVER.GAMMA = gamma

    # Test
    cfg.TEST.EVAL_PERIOD = 100
    cfg.TEST.DETECTIONS_PER_IMAGE = 80

    cfg.MODEL.WEIGHTS = "model_final_f10217.pkl"  # initialize from model zoo
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = batch_size
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # 1 class (hail)
    cfg.MODEL.MASK_ON = True
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.05

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = CocoTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()
    
    #output_dir = cfg.OUTPUT_DIR + '/eval/'
    #os.makedirs(output_dir, exist_ok=True)
    #evaluator = COCOEvaluator("val_hail", cfg, False, output_dir=output_dir)
    #val_loader = build_detection_test_loader(cfg, "val_hail")
    #result = inference_on_dataset(trainer.model, val_loader, evaluator)
    
    #AP_bbox = result['bbox']['AP']
    #AP50_bbox = result['bbox']['AP50']
    #AP75_bbox = result['bbox']['AP75']
    #AP_segm = result['segm']['AP']
    #AP50_segm = result['segm']['AP50']
    #AP75_segm = result['segm']['AP75']
    #return AP_bbox, AP50_bbox, AP75_bbox, AP_segm, AP50_segm, AP75_segm

def run(run_dir, hparams, base_lr, gamma, batch_size):
    """
    Run the hyperparameter tuning experiment and log results to TensorBoard.

    Args:
    - run_dir (str): Directory to save the experiment results.
    - hparams (dict): Hyperparameters for the experiment.
    - base_lr (float): Learning rate.
    - gamma (float): Learning rate decay factor.
    - batch_size (int): Batch size per image.
    """
    with tf.summary.create_file_writer(run_dir).as_default():
        hp.hparams(hparams)  # record the values used in this trial
        train_hail_model(run_dir, base_lr, gamma, batch_size)

class CocoTrainer(DefaultTrainer):
  """
  Custom trainer class for COCO evaluation during training.
  """
  @classmethod
  def build_evaluator(cls, cfg, dataset_name, output_folder=cfg.OUTPUT_DIR):
      return COCOEvaluator(dataset_name, cfg, False, output_folder)

  def build_hooks(self):
    hooks = super().build_hooks()
    hooks.insert(-1,LossEvalHook(
        cfg.TEST.EVAL_PERIOD,
        self.model,
        build_detection_test_loader(
            self.cfg,
            self.cfg.DATASETS.TEST[0],
            DatasetMapper(self.cfg,True)
        )
    ))
    return hooks

class LossEvalHook(HookBase):
    """
    Custom hook for evaluating loss during training.
    """
    def __init__(self, eval_period, model, data_loader):
        self._model = model
        self._period = eval_period
        self._data_loader = data_loader
    
    def _do_loss_eval(self):
        # Copying inference_on_dataset from evaluator.py
        total = len(self._data_loader)
        num_warmup = min(5, total - 1)
            
        start_time = time.perf_counter()
        total_compute_time = 0
        losses = []
        for idx, inputs in enumerate(self._data_loader):            
            if idx == num_warmup:
                start_time = time.perf_counter()
                total_compute_time = 0
            start_compute_time = time.perf_counter()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_compute_time += time.perf_counter() - start_compute_time
            iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
            seconds_per_img = total_compute_time / iters_after_start
            if idx >= num_warmup * 2 or seconds_per_img > 5:
                total_seconds_per_img = (time.perf_counter() - start_time) / iters_after_start
                eta = datetime.timedelta(seconds=int(total_seconds_per_img * (total - idx - 1)))
                log_every_n_seconds(
                    logging.INFO,
                    "Loss on Validation  done {}/{}. {:.4f} s / img. ETA={}".format(
                        idx + 1, total, seconds_per_img, str(eta)
                    ),
                    n=5,
                )
            loss_batch = self._get_loss(inputs)
            losses.append(loss_batch)
        mean_loss = np.mean(losses)
        self.trainer.storage.put_scalar('validation_loss', mean_loss)
        comm.synchronize()

        return losses
            
    def _get_loss(self, data):
        # How loss is calculated on train_loop 
        metrics_dict = self._model(data)
        metrics_dict = {
            k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else float(v)
            for k, v in metrics_dict.items()
        }
        total_losses_reduced = sum(loss for loss in metrics_dict.values())
        return total_losses_reduced
             
    def after_step(self):
        next_iter = self.trainer.iter + 1
        is_final = next_iter == self.trainer.max_iter
        if is_final or (self._period > 0 and next_iter % self._period == 0):
            self._do_loss_eval()
        self.trainer.storage.put_scalars(timetest=12)
        
if __name__=="__main__":
    main()
