#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Sept 8 01:34:25 2018
@author: Arpan
@Description: Utils file to extract frame from folder videos and dump to disk.
Feature : Frames converted to numpy files, easy to subset consecutive frames 
and feed them to the deep network.
"""

import os
import numpy as np
import cv2
import time
import pandas as pd
from joblib import Parallel, delayed
    

def extract_vid_frames(srcFolderPath, destFolderPath, stop='all'):
    """
    Function to extract the features from a list of videos
    
    Parameters:
    ------
    srcFolderPath: str
        path to folder which contains the videos
    destFolderPath: str
        path to store the optical flow values in .bin files
    grid_size: int
        distance between two neighbouring pixel optical flow values.
    stop: str
        to traversel 'stop' no of files in each subdirectory.
    
    Returns: 
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
    batch = 10  # No. of videos in a single batch
    njobs = 5   # No. of threads
    
    for i in range(nrows/batch):
        #batch_diffs = getHOGVideo(filenames_df['infiles'][i], hog)
        # 
        batch_diffs = Parallel(n_jobs=njobs)(delayed(getFrames) \
                          (filenames_df['infiles'][i*batch+j]) \
                          for j in range(batch))
        print "i = "+str(i)
        # Writing the diffs in a serial manner
        for j in range(batch):
            if batch_diffs[j] is not None:
                #with open(filenames_df['outfiles'][i*batch+j] , "wb") as fp:
                #    pickle.dump(batch_diffs[j], fp)
                np.save(filenames_df['outfiles'][i*batch+j], batch_diffs[j])
                print "Written "+str(i*batch+j+1)+" : "+ \
                                    filenames_df['outfiles'][i*batch+j]
            
    # For last batch which may not be complete, extract serially
    last_batch_size = nrows - ((nrows/batch)*batch)
    if last_batch_size > 0:
        batch_diffs = Parallel(n_jobs=njobs)(delayed(getFrames) \
                              (filenames_df['infiles'][(nrows/batch)*batch+j]) \
                              for j in range(last_batch_size)) 
        # Writing the diffs in a serial manner
        for j in range(last_batch_size):
            if batch_diffs[j] is not None:
#                with open(filenames_df['outfiles'][(nrows/batch)*batch+j] , "wb") as fp:
#                    pickle.dump(batch_diffs[j], fp)
                np.save(filenames_df['outfiles'][(nrows/batch)*batch+j], batch_diffs[j])
                print "Written "+str((nrows/batch)*batch+j+1)+" : "+ \
                                    filenames_df['outfiles'][(nrows/batch)*batch+j]
    
    ###########################################################################
    print len(batch_diffs)
    return traversed


def getTotalFramesVid(srcVideoPath):
    """
    Return the total number of frames in the video
    
    Parameters:
    ------
    srcVideoPath: str
        complete path of the source input video file
        
    Returns:
    ------
    total frames present in the given video file
    """
    cap = cv2.VideoCapture(srcVideoPath)
    # if the videoCapture object is not opened then exit without traceback
    if not cap.isOpened():
        print("Error reading the video file !!")
        return 0

    tot_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    return tot_frames    


def getFrames(srcVideoPath):
    """
    Function to read all the frames of the video file into a single numpy matrix
    and return that matrix to the caller. This function can be called parallely 
    based on the amount of memory available.
    """
    # get the VideoCapture object
    cap = cv2.VideoCapture(srcVideoPath)
    
    # if the videoCapture object is not opened then exit without traceback
    if not cap.isOpened():
        print("Error reading the video file !!")
        return None
    
    w, h = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    #totalFrames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    frameCount = 0
    features_current_file = []
    
    #ret, prev_frame = cap.read()
    assert cap.isOpened(), "Capture object does not return a frame!"
    #prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    
    # Iterate over the entire video to get the optical flow features.
    while(cap.isOpened()):
        frameCount +=1
        ret, curr_frame = cap.read()
        if not ret:
            break
        curr_frame = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        # taking centre crop of 112 x 112
        curr_frame = curr_frame[(w/2-56):(w/2+56), (h/2-56):(h/2+56)]
        #flow = cv2.calcOpticalFlowFarneback(prev_frame,curr_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        #hog_feature = hog.compute(curr_frame)   # vector
        # saving as a list of float values (after converting into 1D array)
        #features_current_file.append(hog_feature.ravel().tolist())
        features_current_file.append(curr_frame)

    # When everything done, release the capture
    cap.release()
    #print "{}/{} frames in {}".format(frameCount, totalFrames, srcVideoPath)
    #return features_current_file
    return np.array(features_current_file)      # convert to N x w x h


if __name__=='__main__':
    # The srcPath should have subfolders that contain the training, val, test videos.

    #srcPath = '/home/arpan/DATA_Drive/Cricket/dataset_25_fps'
    srcPath = "/home/hadoop/VisionWorkspace/VideoData/sample_cricket/ICC WT20"
    destPath = "/home/hadoop/VisionWorkspace/Cricket/localization_rnn/numpy_frames_cropped"
    if not os.path.exists(srcPath):
        srcPath = "/opt/datasets/cricket/ICC_WT20"
        destPath = "/home/arpan/VisionWorkspace/localization_rnn/numpy_frames_cropped"
    
    start = time.time()
    extract_vid_frames(srcPath, destPath, stop='all')
    end = time.time()
    print "Total execution time : "+str(end-start)