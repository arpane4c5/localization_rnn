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

from torch.autograd import Variable

from glob import glob

import skimage.io as io
from skimage.transform import resize


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



def get_sport_clip(clip_name, verbose=True):
    """
    Loads a clip to be fed to C3D for classification.
    TODO: should I remove mean here?
    
    Parameters
    ----------
    clip_name: str
        the name of the clip (subfolder in 'data').
    verbose: bool
        if True, shows the unrolled clip (default is True).

    Returns
    -------
    Tensor
        a pytorch batch (n, ch, fr, h, w).
    """

    clip = sorted(glob(os.path.join('data', clip_name, '*.jpg')))
    clip = np.array([resize(io.imread(frame), output_shape=(112, 200), preserve_range=True) for frame in clip])
    clip = clip[:, :, 44:44+112, :]  # crop centrally

    if verbose:
        clip_img = np.reshape(clip.transpose(1, 0, 2, 3), (112, 16 * 112, 3))
        io.imshow(clip_img.astype(np.uint8))
        io.show()

    clip = clip.transpose(3, 0, 1, 2)  # ch, fr, h, w
    clip = np.expand_dims(clip, axis=0)  # batch axis
    clip = np.float32(clip)

    return torch.from_numpy(clip)


def read_labels_from_file(filepath):
    """
    Reads Sport1M labels from file
    
    Parameters
    ----------
    filepath: str
        the file.
        
    Returns
    -------
    list
        list of sport names.
    """
    with open(filepath, 'r') as f:
        labels = [line.strip() for line in f.readlines()]
    return labels



if __name__=='__main__':
    
    # create a model 
    import model_c3d as c3d
    
    model = c3d.C3D()
    
    ###########################################################################
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

    ###########################################################################
    
    # load a clip to be predicted
    X = get_sport_clip('TaiChi/v_TaiChi_g18_c01', verbose=False)
    X = Variable(X)
    #X = X.cuda()

    # get network pretrained model
    model.load_state_dict(torch.load('c3d.pickle'))
    #model.cuda()
    model.eval()

    # perform prediction
    prediction = model(X)
    prediction = prediction.data.cpu().numpy()

    # read labels
    labels = read_labels_from_file('labels.txt')

#    # print top predictions
#    top_inds = prediction[0].argsort()[::-1][:5]  # reverse sort and take five largest items
#    print('\nTop 5:')
#    for i in top_inds:
#        print('{:.5f} {}'.format(prediction[0][i], labels[i]))


