#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 13 19:36:05 2018

@author: Arpan
"""

import torch
import numpy as np
import cv2
import os
import pickle
import shutil
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable


def create_variable(tensor):
    # Do cuda() before wrapping with variable
    if torch.cuda.is_available():
        return Variable(tensor.cuda())
    else:
        return Variable(tensor)

# Split the dataset files into training, validation and test sets
# All video files present at the same path (assumed)
def split_dataset_files(datasetPath):
    filenames = sorted(os.listdir(datasetPath))         # read the filename
    filenames = [t.split('.')[0] for t in filenames]   # remove the extension
    return filenames[:16], filenames[16:21], filenames[21:]
    

# function to count the number of parameters in the model
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# send a list of lists containing start and end frames of actions
# eg [98, 218], [376, 679], [2127, 2356], [4060, 4121], [4137, 4250]]
# Return a sequence of action labels corresponding to frame features
def get_vid_labels_vec(labels, vid_len):
    if labels is None or len(labels)==0:
        return []
    v = []
    for i,x in enumerate(labels):
        if i==0:
            v.extend([0]*(x[0]-1))
        else:
            v.extend([0]*(x[0]-labels[i-1][1]-1))
        v.extend([1]*(x[1]-x[0]+1))  
    v.extend([0]*(vid_len - labels[-1][1]))
    return v

# return the number of frames present in a video vid
def getNFrames(vid):
    cap = cv2.VideoCapture(vid)
    if not cap.isOpened():
        import sys
        print "Capture Object not opened ! Abort"
        sys.exit(0)
        
    l = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return l


def getFeatureVectors(datasetpath, videoFiles, sequences):
    """
     Iteratively take the batch information and extract the feature sequences 
     from the videos
     datasetpath : Prefix of the path to the dataset containing the videos
     videoFiles : list/tuple of filenames for the videos (size n)
     sequences :  list of start frame numbers and end frame numbers 
     sequences[0] and [1] are torch.LongTensor of size n each.
     returns a list of lists. Inner list contains a sequence of arrays 
    """
    grid_size = 20
    batch_feats = []
    # Iterate over the videoFiles in the batch and extract the corresponding feature
    for i, videoFile in enumerate(videoFiles):
        videoFile = videoFile.split('/')[1]
        vid_feat_seq = []
        # use capture object to get the sequences
        cap = cv2.VideoCapture(os.path.join(datasetpath, videoFile))
        if not cap.isOpened():
            print "Capture object not opened : {}".format(videoFile)
            import sys
            sys.exit(0)
            
        start_frame = sequences[0][i]
        end_frame = sequences[1][i]
        ####################################################    
        #print "Start Times : {} :: End Times : {}".format(start_frame, end_frame)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        ret, prev_frame = cap.read()
        if ret:
            # convert frame to GRAYSCALE
            prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        else:
            print "Frame not read: {} : {}".format(videoFile, start_frame)

        for stime in range(start_frame+1, end_frame+1):
            ret, frame = cap.read()
            if not ret:
                print "Frame not read : {} : {}".format(videoFile, stime)
                continue
            
            curr_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            #cv2.calcOpticalFlowFarneback(prev, next, pyr_scale, levels, winsize, 
            #                iterations, poly_n, poly_sigma, flags[, flow])
            # prev(y,x)~next(y+flow(y,x)[1], x+flow(y,x)[0])
            flow = cv2.calcOpticalFlowFarneback(prev_frame,curr_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            #print "For frames: ("+str(stime-1)+","+str(stime)+") :: shape : "+str(flow.shape)
            
            mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
            # stack sliced arrays along the first axis (2, 12, 16)
            sliced_flow = np.stack(( mag[::grid_size, ::grid_size], \
                                    ang[::grid_size, ::grid_size]), axis=0)
            sliced_flow = sliced_flow.ravel()   # flatten
            vid_feat_seq.append(sliced_flow.tolist())    # append to list
            prev_frame = curr_frame
        cap.release()            
        batch_feats.append(vid_feat_seq)
        
    return batch_feats
    

def readAllOFfeatures(OFfeaturesPath, keys):
    """
    Load the features of the train/val/test set into a dictionary. Dictionary 
    has key as the filename(without ext) of video and value as the numpy feature 
    matrix.
    """
    feats = {}
    for k in keys:
        featpath = os.path.join(OFfeaturesPath, k)+".bin"
        assert os.path.exists(featpath), "featpath not found {}".format(featpath)
        with open(featpath, "rb") as fobj:
            feats[k] = pickle.load(fobj)
            
    print "Features loaded into dictionary ..."
    return feats

def readAllHOGfeatures(HOGfeaturesPath, keys):
    """
    Load the features of the train/val/test set into a dictionary. Dictionary 
    has key as the filename(without ext) of video and value as the numpy feature 
    matrix.
    """
    feats = {}
    for k in keys:
        featpath = os.path.join(HOGfeaturesPath, k)+".bin"
        with open(featpath, "rb") as fobj:
            feats[k] = pickle.load(fobj)
            
    print "Features loaded into dictionary ..."
    return feats
        
def readAllNumpyFrames(numpyFramesPath, keys):
    """
    Load the frames of the train/val/test set into a dictionary. Dictionary 
    has key as the filename(without ext) of video and value as the Nxhxw numpy  
    matrix.
    """
    feats = {}
    for k in keys:
        featpath = os.path.join(numpyFramesPath, k)+".npy"
        feats[k] = np.load(featpath)
            
    print "Features loaded into dictionary ..."
    return feats

    
def getFeatureVectorsFromDump(features, videoFiles, sequences, motion=True):
    """Select only the batch features from the dictionary of features (corresponding
    to the given sequences) and return them as a list of lists. 
    OFfeatures: a dictionary of features {vidname: numpy matrix, ...}
    videoFiles: the list of filenames for a batch
    sequences: the start and end frame numbers in the batch videos to be sampled.
    """
    #grid_size = 20
    batch_feats = []
    # Iterate over the videoFiles in the batch and extract the corresponding feature
    for i, videoFile in enumerate(videoFiles):
        # get key value for the video. Use this to read features from dictionary
        videoFile = videoFile.split('/')[1].rsplit('.', 1)[0]
            
        start_frame = sequences[0][i]   # starting point of sequences in video
        end_frame = sequences[1][i]     # end point
        # Load features
        # (N-1) sized list of vectors of 1152 dim
        vidFeats = features[videoFile]  
        if motion:
            vid_feat_seq = vidFeats[start_frame:end_frame]
        else:
            vid_feat_seq = vidFeats[start_frame:(end_frame+1)]
        
        batch_feats.append(vid_feat_seq)
        
    return batch_feats

def getC3DFeatures(c3d_model, frames, videoFiles, sequences):
    """
    Select the batch frames from the dictionary of numpy frames (corresponding
    to the given sequences) and extract C3D feature vector from them (fc7 layer)
    return them as a list of lists. 
    OFfeatures: a dictionary of features {vidname: numpy matrix, ...}
    videoFiles: the list of filenames for a batch
    sequences: the start and end frame numbers in the batch videos to be sampled.
    """
    #grid_size = 20
    batch_feats = []
    # Iterate over the videoFiles in the batch and extract the corresponding feature
    for i, videoFile in enumerate(videoFiles):
        # get key value for the video. Use this to read features from dictionary
        videoFile = videoFile.split('/')[1].rsplit('.', 1)[0]
            
        start_frame = sequences[0][i]   # starting point of sequences in video
        end_frame = sequences[1][i]     # end point
        # Load features
        # (N-1) sized list of vectors of 1152 dim
        vidFeats = features[videoFile]  
        
        vid_frames_seq = vidFeats[start_frame:(end_frame+1)]
        
        batch_feats.append(vid_frames_seq)
        
    return batch_feats

# Inputs: feats: list of lists
def make_variables(feats, labels, motion=True):
    # Create the input tensors and target label tensors
    #for item in feats:
        # item is a list with (sequence of) 9 1D vectors (each of 1152 size)
        
    feats = torch.Tensor(feats)
    feats[feats==float("-Inf")] = 0
    feats[feats==float("Inf")] = 0
    # Form the target labels 
    target = []
#    for i in range(labels[0].size(0)):
#        if labels[0][i]<5:
#            target.append(0)
#        else:
#            target.append(1)
            
    for i in range(labels[0].size(0)):
        if labels[0][i]>0:
            if motion:
                target.extend([0]*(labels[0][i]-1) + [1]*labels[1][i])
            else:
                target.extend([0]*labels[0][i] + [1]*labels[1][i])
        else:
            if motion:
                target.extend([0]*labels[0][i] + [1]*(labels[1][i]-1))
            else:
                target.extend([0]*labels[0][i] + [1]*labels[1][i])
    # Form a wrap into a tensor variable as B X S X I
    return create_variable(feats), create_variable(torch.Tensor(target))

def save_checkpoint(state, is_best, save_path, filename):
    filename = os.path.join(save_path, filename)
    torch.save(state, filename)
    if is_best:
        bestname = os.path.join(save_path, 'model_best.pth.tar')
        shutil.copyfile(filename, bestname)


# function to remove the action segments that have less than "epsilon" frames.
def filter_action_segments(shots_dict, epsilon=10):
    filtered_shots = {}
    for k,v in shots_dict.iteritems():
        vsegs = []
        for segment in v:
            if (segment[1]-segment[0] >= epsilon):
                vsegs.append(segment)
        filtered_shots[k] = vsegs
    return filtered_shots

# function to remove the non-action segments that have less than "epsilon" frames.
# Here we need to merge one or more action segments
def filter_non_action_segments(shots_dict, epsilon=10):
    filtered_shots = {}
    for k,v in shots_dict.iteritems():
        vsegs = []
        isFirstSeg = True
        for segment in v:
            if isFirstSeg:
                prev_st, prev_end = segment
                isFirstSeg = False
                continue
            if (segment[0] - prev_end) <= epsilon:
                prev_end = segment[1]   # new end of segment
            else:       # Append to list
                vsegs.append((prev_st, prev_end))
                prev_st, prev_end = segment     
        # For last segment
        if len(v) > 0:      # For last segment
            vsegs.append((prev_st, prev_end))
        filtered_shots[k] = vsegs
    return filtered_shots