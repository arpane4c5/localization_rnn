#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed May 23 00:20:54 2018

@author: Arpan
Description: Extract frames from the UCF101 dataset videos

Generate a list of the files to process. We're going to use the images that ship with caffe.
$ find /opt/C3D/C3D-v1.1/examples/images -type f -exec echo {} \; > examples/_temp/temp.txt

The ImageDataLayer we'll use expects labels after each filenames, 
so let's add a 0 to the end of each line
$ sed "s/$/ 0/" examples/_temp/temp.txt > examples/_temp/file_list.txt

"""


import cv2
import os

# Local system
# contains subfolders and videos inside the subfolders
#DATASET_PATH = "/home/hadoop/VisionWorkspace/VideoData/UCF/UCF-101/"
DATASET_PATH = "/home/hadoop/VisionWorkspace/VideoData/sample_cricket/ICC WT20"
#DESTPATH = "/home/hadoop/VisionWorkspace/VideoData/UCF/frm"
DESTPATH = "/home/hadoop/VisionWorkspace/VideoData/sample_cricket/frms"

# server 
if os.path.exists("/home/arpan/DATA_Drive/video_datasets/UCF/UCF-101"):
    DATASET_PATH = "/home/arpan/DATA_Drive/video_datasets/UCF/UCF-101"
    DESTPATH = "/home/arpan/VisionWorkspace/c3dFeatureExtraction/UCF/frm"

def extract_frames(srcVid, destPath, vid_name):
    prefix = vid_name.rsplit(".", 1)[0]
    if not os.path.exists(destPath):
        os.makedirs(os.path.join(destPath, prefix))
    
    cap = cv2.VideoCapture(os.path.join(srcVid, vid_name))
    if not cap.isOpened():
        print "Cannot open video. Returning !!"
        return
    
    count = 1
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            #if (count-1)%16==0:
            cv2.imwrite(os.path.join(destPath, prefix, "{:06}.jpg".format(count)), frame)
            print "Written frame {}".format(count)
        else:
            break
        count+=1
        
    cap.release()

if __name__ == '__main__':
    
    actions_list = os.listdir(DATASET_PATH)
    print actions_list
    
    nacts = 0
    
#    for sf in actions_list:
#        vids_list = os.listdir(os.path.join(DATASET_PATH, sf))
#        for video in vids_list:
#            extract_frames(os.path.join(DATASET_PATH, sf), \
#                       os.path.join(DESTPATH, sf), video)
#            break
#        nacts +=1
#        if nacts==2:
#            break
    
    # Extracting from one cricket sample video
    for video in actions_list:
        extract_frames(DATASET_PATH, DESTPATH, video)
        break
        nacts +=1
        if nacts==2:
            break

    print "Done"
