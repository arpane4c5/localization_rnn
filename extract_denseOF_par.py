#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat June 30 01:34:25 2018

@author: Arpan

@Description: Utils file to extract Farneback dense optical flow features 
from folder videos and dump to disk.

Feature : Farneback Dense Optical Flow: Magnitudes and Angles (with grid_size)
Execution Time: 1365.583 secs (Njobs=10, batch=10) (nVids = 26)
Execution Time: 1420.7 secs (Njobs=10, batch=10) (nVids=26) grid=40

"""

import os
import numpy as np
import cv2
import time
import pandas as pd
from joblib import Parallel, delayed

def extract_dense_OF_vids(srcFolderPath, destFolderPath, grid_size=20, njobs=1, batch=10, stop='all'):
    """
    Function to extract the features from a list of videos, given the path of the
    videos and the destination path for the features
    
    Parameters:
    ------
    srcFolderPath: str
        path to folder which contains the videos
    destFolderPath: str
        path to store the optical flow values in .npy files
    grid_size: int
        distance between two neighbouring pixel optical flow values.
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
            outfiles.append(os.path.join(destFolderPath, vid.rsplit('.',1)[0]+".npy"))
            nFrames.append(getTotalFramesVid(os.path.join(srcFolderPath, vid)))
            # save at the destination, if extracted successfully
            traversed += 1
                   
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
        # 
        batch_diffs = Parallel(n_jobs=njobs)(delayed(getFarnebackOFVideo) \
                          (filenames_df['infiles'][i*batch+j], grid_size) \
                          for j in range(batch))
        
        # Writing the files to the disk in a serial manner
        for j in range(batch):
            if batch_diffs[j] is not None:
                np.save(filenames_df['outfiles'][i*batch+j], batch_diffs[j])
                print "Written "+str(i*batch+j+1)+" : "+ \
                                    filenames_df['outfiles'][i*batch+j]
            
    # For last batch which may not be complete, extract serially
    last_batch_size = nrows - ((nrows/batch)*batch)
    if last_batch_size > 0:
        batch_diffs = Parallel(n_jobs=njobs)(delayed(getFarnebackOFVideo) \
                              (filenames_df['infiles'][(nrows/batch)*batch+j], grid_size) \
                              for j in range(last_batch_size)) 
        # Writing the files to the disk in a serial manner
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


def getFarnebackOFVideo(srcVideoPath, grid_size):
    """
    Function to get the Farneback dense optical flow features for the video file
    and sample magnitude and angle features with a distance of grid_size between 
    two neighbours.
    Copied and editted from shot_detection.py script
    
    Parameters:
    ------
    srcVideoPath: str
        complete path of a video file
    grid_size: int
        distance between two consecutive sampling pixels.
    
    Returns:
    ------
    np.ndarray of dimension (N-1, 2 x (360/grid_size) x (640/grid_size))
    """
    # get the VideoCapture object
    cap = cv2.VideoCapture(srcVideoPath)
    
    # if the videoCapture object is not opened then return None
    if not cap.isOpened():
        print("Error reading the video file !!")
        return None
    
#    dimensions = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), \
#                  int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
#    totalFrames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    frameCount = 0
    features_current_file = []
    
    ret, prev_frame = cap.read()
    assert ret, "Capture object does not return a frame!"
    prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    
    # Iterate over the entire video to get the optical flow features.
    while(cap.isOpened()):
        frameCount +=1
        ret, curr_frame = cap.read()
        if not ret:
            break
        curr_frame = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev_frame,curr_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
        # stack sliced arrays along the first axis (2, (360/grid), (640/grid))
        sliced_flow = np.stack(( mag[::grid_size, ::grid_size], \
                                ang[::grid_size, ::grid_size]), axis=0)
        
        # For extracting only magnitude features, uncomment the following
        #sliced_flow = mag[::grid_size, ::grid_size]

        #feature.append(sliced_flow[..., 0].ravel())
        #feature.append(sliced_flow[..., 1].ravel())
        # saving as a list of float values (after converting into 1D array)
        #features_current_file.append(sliced_flow.ravel().tolist())   #slow at load time
        # convert to (1, 2x(H/grid)x(W/grid)) matrix.
        sliced_flow = np.expand_dims(sliced_flow.flatten(), axis=0)
        features_current_file.append(sliced_flow)
        prev_frame = curr_frame

    cap.release()
    #print "{}/{} frames in {}".format(frameCount, totalFrames, srcVideoPath)
    return np.array(features_current_file)  # (N-1, 1, 2x(H/grid)x(W/grid))


if __name__=='__main__':
    
    gridSize = 40
    batch = 10  # No. of videos in a single batch
    njobs = 10   # No. of threads
    # Server params
    srcPath = '/opt/datasets/cricket/ICC_WT20'
    destPath = "/home/arpan/VisionWorkspace/localization_rnn/OF_npy_grid"+str(gridSize)
    # localhost params
    if not os.path.exists(srcPath):
        srcPath = "/home/hadoop/VisionWorkspace/VideoData/sample_cricket/ICC WT20"
        destPath = "/home/hadoop/VisionWorkspace/Cricket/localization_rnn/OF_npy_grid"+str(gridSize)
        
    start = time.time()
    extract_dense_OF_vids(srcPath, destPath, gridSize, njobs, batch, stop='all')
    end = time.time()
    print "Total execution time : "+str(end-start)
    