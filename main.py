#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 21 13:53:27 2018

@author: Arpan

@Description: C3D model implementation in PyTorch. Learn cricket strokes.
"""

import torch
import numpy as np
import cv2
import os

# Local Paths
LABELS = "/home/hadoop/VisionWorkspace/Cricket/scripts/supporting_files/sample_set_labels/sample_labels_shots/ICC WT20"
DATASET = "/home/hadoop/VisionWorkspace/VideoData/sample_cricket/ICC WT20"

# Server Paths
if os.path.exists("/opt/datasets/cricket/ICC_WT20"):
    LABELS = "/home/arpan/VisionWorkspace/shot_detection/supporting_files/sample_set_labels/sample_labels_shots/ICC WT20"
    DATASET = "/opt/datasets/cricket/ICC_WT20"


def train_model(model, destpath):
    train_files = sorted(os.listdir(DATASET))
    labels = sorted(os.listdir(LABELS))
    
    print train_files
    print labels

# function to count the number of parameters in the model
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__=='__main__':
    
    # create a model 
    import model_c3d as c3d
    
    model = c3d.C3D()
    
    #print model
    train_model(model, "data")
    
    # count no. of parameters in the model
    #model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    #params = sum([np.prod(p.size()) for p in model_parameters])
    # or call count_paramters(model)  
    print "#Parameters : {} ".format(count_parameters(model))
    
    # Creation of a training set, validation set, test set meta info file.
    
    # Train the model
    
    # 
    