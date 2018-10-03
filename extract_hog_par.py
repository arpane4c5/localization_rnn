#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat June 30 01:34:25 2018
@author: Arpan
@Description: Utils file to extract HOG features from folder videos and dump 
to disk.(Parallelized Version)
Feature : Histogram of Oriented Gradients on 64x64 centre crops. HOG winSize=64,
Blocksize=32, BlockStride=8, cellsize=8, nbins=9 (in hog.xml). (Vec Size=3600)
Execution Time: 212.55 secs (njobs=1, batch=4) 64x64 centre crops 
                117.88 secs (njobs=2, batch=4)   "
"""

import os
import numpy as np
import cv2
import time
import pandas as pd
from joblib import Parallel, delayed

def extract_hog_vids(srcFolderPath, destFolderPath, hog, njobs=1, batch=10, stop='all'):
    """
    Function to extract the HOG features from a list of videos, parallely.
    
    Parameters: 
    ------
    srcFolderPath: str
        complete path to folder which contains the videos
    destFolderPath: str
        path to store the optical flow values in .npy files
    hog: str
        path to .xml file describing the HOG parameters to be used.
    njobs: int
        no. of cores to be used parallely
    batch: int
        no. of video files in a batch. A batch executed parallely and 
        is dumped to disk before starting another batch. Depends on RAM.
    stop: str or int(if to be stopped after some files)
        to traversel 'stop' no of files in each subdirectory.
    
    Return: 
    ------
    traversed: int
        no of videos traversed successfully
    """
    # iterate over the subfolders in srcFolderPath and extract for each video 
    vfiles = os.listdir(srcFolderPath)
    
    infiles, outfiles, nFrames = [], [], []
    
    traversed = 0
    # create destination path to store the files
    if not os.path.exists(destFolderPath):
        os.makedirs(destFolderPath)
            
    # iterate over the video files inside the directory sf
    for vid in vfiles:
        if os.path.isfile(os.path.join(srcFolderPath, vid)) and vid.rsplit('.', 1)[1] in {'avi', 'mp4'}:
            infiles.append(os.path.join(srcFolderPath, vid))
            outfiles.append(os.path.join(destFolderPath, vid.rsplit('.', 1)[0]+".npy"))
            nFrames.append(getTotalFramesVid(os.path.join(srcFolderPath, vid)))
            # save at the destination, if extracted successfully
            traversed += 1
#            print "Done "+str(traversed_tot+traversed)+" : "+sf+"/"+vid
                    
                # to stop after successful traversal of 2 videos, if stop != 'all'
            if stop != 'all' and traversed == stop:
                break
                    
    print "No. of files to be written to destination : "+str(traversed)
    if traversed == 0:
        print "Check the structure of the dataset folders !!"
        return traversed
    ###########################################################################
    #### Form the pandas Dataframe and parallelize over the files.
    filenames_df = pd.DataFrame({"infiles":infiles, "outfiles": outfiles, "nframes": nFrames})
    filenames_df = filenames_df.sort_values(["nframes"], ascending=[True])
    filenames_df = filenames_df.reset_index(drop=True)
    nrows = filenames_df.shape[0]
    
    for i in range(nrows/batch):
        #batch_diffs = getHOGVideo(filenames_df['infiles'][i], hog)
        # 
        batch_diffs = Parallel(n_jobs=njobs)(delayed(getHOGVideo) \
                          (filenames_df['infiles'][i*batch+j], hog) \
                          for j in range(batch))
        print "i = "+str(i)
        # Writing the diffs in a serial manner
        for j in range(batch):
            if batch_diffs[j] is not None:
                np.save(filenames_df['outfiles'][i*batch+j], batch_diffs[j])
                print "Written "+str(i*batch+j+1)+" : "+ \
                                    filenames_df['outfiles'][i*batch+j]
            
    # For last batch which may not be complete, extract serially
    last_batch_size = nrows - ((nrows/batch)*batch)
    if last_batch_size > 0:
        batch_diffs = Parallel(n_jobs=njobs)(delayed(getHOGVideo) \
                              (filenames_df['infiles'][(nrows/batch)*batch+j], hog) \
                              for j in range(last_batch_size)) 
        # Writing the diffs in a serial manner
        for j in range(last_batch_size):
            if batch_diffs[j] is not None:
                np.save(filenames_df['outfiles'][(nrows/batch)*batch+j], batch_diffs[j])
                print "Written "+str((nrows/batch)*batch+j+1)+" : "+ \
                                    filenames_df['outfiles'][(nrows/batch)*batch+j]
    
    ###########################################################################
    return traversed

def getTotalFramesVid(srcVideoPath):
    """
    Return the total number of frames in the video
    Parameter:
    ------
    srcVideoPath: str
        complete path to the video file
        
    Returns: 
    ------
    tot_frames: int
        total number of frames in the video file.
    """
    cap = cv2.VideoCapture(srcVideoPath)
    # if the videoCapture object is not opened then return 0 frames
    if not cap.isOpened():
        print("Error reading the video file !!")
        return 0

    tot_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    return tot_frames

def getHOGVideo(srcVideoPath, hogPath):
    """
    Function to get the HOG features for all the frames of a video file. 
    The HOG parameters are defined in the hog.xml file. WinSize=64, BlockSize=32,
    blockStride=8, cellSize=8 and nbins=9.
    Copied and editted from shot_detection.py script
    
    Parameters: 
    ------
    srcVideoPath: str
        complete path to a single video file.
    hog: str
        path to the HOG parameters file.
        
    Returns: 
    ------
    features_current_file: np.ndarray
    
    """
    # get the VideoCapture object
    cap = cv2.VideoCapture(srcVideoPath)
    
    # if the videoCapture object is not opened then exit without traceback
    if not cap.isOpened():
        print("Error reading the video file !!")
        return None
    
    w, h = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), \
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    #totalFrames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    frameCount = 0
    features_current_file = []
    #ret, prev_frame = cap.read()
    assert cap.isOpened(), "Capture object does not return a frame!"
    #prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    
    # Iterate over the entire video to get the optical flow features.
    while cap.isOpened():
        frameCount += 1
        ret, curr_frame = cap.read()
        if not ret:
            break
        curr_frame = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        # Take the centre crop of the frames (64 x 64)
        curr_frame = curr_frame[(w/2-32):(w/2+32), (h/2-32):(h/2+32)]
        # compute the HOG feature vector 
        hog = cv2.HOGDescriptor(hogPath)        # get cv2.HOGDescriptor object
        hog_feature = hog.compute(curr_frame)   # get 3600 x 1 matrix (not vec)
        # saving as a list of float matrices (dim 1 x vec_size)
        hog_feature = np.expand_dims(hog_feature.flatten(), axis = 0)
        features_current_file.append(hog_feature)
        #features_current_file.append(hog_feature.ravel().tolist())

    cap.release()
    #print "{}/{} frames in {}".format(frameCount, totalFrames, srcVideoPath)
    return np.array(features_current_file) # N x 1 x vec_size


if __name__ == '__main__':
    # For > 1 jobs, Pickling error due to call to Parallel and cannot serialize
    batch = 4  # No. of videos in a single batch
    njobs = 2   # No. of threads
    
    # Server params
    srcPath = '/opt/datasets/cricket/ICC_WT20'
    destPath = "/home/arpan/VisionWorkspace/localization_rnn/hog_feats_64x64"
    
    if not os.path.exists(srcPath):
        srcPath = "/home/hadoop/VisionWorkspace/VideoData/sample_cricket/ICC WT20"
        destPath = "/home/hadoop/VisionWorkspace/Cricket/localization_rnn/hog_feats_64x64"
    
    hog_params_file = "hog.xml"     # in current dir
    #hog = cv2.HOGDescriptor(hog_params_file)   # cannot send cv2.HOGDescriptor
    # results in PicklingError when called for >1 njobs.
    start = time.time()
    extract_hog_vids(srcPath, destPath, hog_params_file, njobs, batch, stop='all')
    end = time.time()
    print "Total execution time : "+str(end-start)
    