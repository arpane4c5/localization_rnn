#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 18 21:56:15 2018

@author: Arpan

@Description: Take predictions json file as input, along with the dataset paths, 
and create a video depicting the output predictions for a set of sample videos.
"""

import json
import os
import cv2

# Local Paths
LABELS = "/home/hadoop/VisionWorkspace/Cricket/scripts/supporting_files/sample_set_labels/sample_labels_shots/ICC WT20"
DATASET = "/home/hadoop/VisionWorkspace/VideoData/sample_cricket/ICC WT20"

# Server Paths
if os.path.exists("/opt/datasets/cricket/ICC_WT20"):
    LABELS = "/home/arpan/VisionWorkspace/shot_detection/supporting_files/sample_set_labels/sample_labels_shots/ICC WT20"
    DATASET = "/opt/datasets/cricket/ICC_WT20"

# Create a video of shot predictions. 
def create_video(gt_dir, shots_dict, destfile = "prediction_vid.avi"):
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    # Check whether all videos exist and then create the output stream
    for i,sf in enumerate(shots_dict.keys()):
        k = ((sf.split("/")[1]).rsplit(".", 1)[0]) + ".json"
        labelfile = os.path.join(gt_dir, k)
        vidfilepath = os.path.join(DATASET, sf.split('/',1)[1])
        assert os.path.exists(labelfile), "Label file does not exist."
        assert os.path.exists(vidfilepath), "Source video file does not exist"
        
        if i==(len(shots_dict.keys())-1):
            # Define the codec and create VideoWriter object
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            cap = cv2.VideoCapture(vidfilepath)
            if not cap.isOpened():
                print("Capture object not opened of last video !! Abort !!")
                return
            fps = cap.get(cv2.CAP_PROP_FPS)
            (w, h) = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), \
                     int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))    
            cap.release()
            
    # Create the output stream 
    out = cv2.VideoWriter(destfile, fourcc, fps, (w/2, h/2))

    # match the val/test keys with the filenames present in the LABELS folder
    for sf in shots_dict.keys():
        k = ((sf.split("/")[1]).rsplit(".", 1)[0]) + ".json"
        labelfile = os.path.join(gt_dir, k)
        vidfilepath = os.path.join(DATASET, sf.split('/',1)[1])
        with open(labelfile, 'r') as fp:
            vid_gt = json.load(fp)
            
        vid_key = vid_gt.keys()[0]      # only one key in dict is saved
        gt_list = vid_gt[vid_key]       # list of tuples [[preFNum, postFNum], ...]
        test_list = shots_dict[vid_key]
        
        currVidCap = cv2.VideoCapture(vidfilepath)
        
        if not currVidCap.isOpened():
            print("Capture object not opened !! Abort !!")
            return
        
        gt_list = get_flags_list(gt_list, int(currVidCap.get(cv2.CAP_PROP_FRAME_COUNT)))
        test_list = get_flags_list(test_list, int(currVidCap.get(cv2.CAP_PROP_FRAME_COUNT)))
        currVidCap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        for i, gt_flag in enumerate(gt_list):
            
            ret, frame = currVidCap.read()
            if ret:
                
                test_flag = test_list[i]
                cv2.putText(frame,'Ground Truth',(470,40), font, 0.5,(255,255,255),1,cv2.LINE_AA)
                if gt_flag:    # 
                    cv2.circle(frame, (600, 30), 15, (0, 255, 0), -1)
                else:
                    cv2.circle(frame, (600, 30), 15, (0, 0, 255), -1)
                
                cv2.putText(frame,'Prediction',(470,80), font, 0.5,(255,255,255),1,cv2.LINE_AA)
                if test_flag:
                    cv2.circle(frame, (600, 70), 15, (0, 255, 0), -1)
                else:
                    cv2.circle(frame, (600, 70), 15, (0, 0, 255), -1)
                frame = cv2.resize(frame, (w/2, h/2), interpolation=cv2.INTER_AREA)          
                out.write(frame)
                #cv2.imshow("Frame", frame)
                #waitTillEscPressed()
                print(k+" : "+str(i))
                    
            else:
                print("Next frame is NULL")
            
    currVidCap.release()
    out.release()
    #cv2.destroyAllWindows()
    
# get the segments list and total length of the video (no. of frames) and form
# a list of binary flags corresponding to the frames of the video
def get_flags_list(segments_list, vidLen):
    flags = []
    currLen = 0    
    for seg in segments_list:
        flags.extend((seg[0]-currLen)*[False])
        flags.extend((seg[1]-seg[0]+1)*[True])
        currLen = currLen + (seg[0]-currLen) + (seg[1]-seg[0]+1)
        
    flags.extend((vidLen-currLen)*[False])    
    assert len(flags)==vidLen, "Count not valid"
    return flags

    
def waitTillEscPressed():
    while(True):
        # For moving forward
        if cv2.waitKey(0)==27:
            print("Esc Pressed. Move Forward.")
            return 1


if __name__ == '__main__':
    # Take the prediction file (json as input)
    pred_shots_file = "predicted_localizations_th0_5_filt60.json"
    with open(pred_shots_file, 'r') as fp:
        shots_dict = json.load(fp)
        
    create_video(LABELS, shots_dict, "predicted_vid_180x320.avi")
