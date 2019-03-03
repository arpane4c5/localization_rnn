#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 24 03:24:10 2018

@author: hadoop
"""

import cv2
import os

H = 360
W = 640

def extract_vid_frames(srcVidPath, destPath, i):
    
    if not os.path.exists(srcVidPath):
        print("Source file does not exist. Abort!!")
        return
    
    if not os.path.exists(destPath):
        os.makedirs(destPath)
        
    cap = cv2.VideoCapture(srcVidPath)
    if not cap.isOpened():
        print("Capture object not opened. Abort!!")
        return
    
    #j = 0
    while cap.isOpened():
        ret, frame = cap.read()
        print("Done : "+str(i))
        if ret:
            if i<=4000:
                cv2.imwrite(os.path.join(destPath, str(i)+".png"), frame)
            else:
                return
        else:
            return
        i+=1
        
        
    print("Done")
    
    
def form_video_from_frames(srcFramesPath, destFileName):
    flist = os.listdir(srcFramesPath)
    flist = [int(f.split('.')[0]) for f in flist]
    flist = sorted(flist)
    print(flist)
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(destFileName, fourcc, 25.0, (W, H))
    
    for f in flist:
        filepath = os.path.join(srcFramesPath, str(f)+'.png')
        frame = cv2.imread(filepath)
        (h, w, c) = frame.shape
        if (h!=H or w!=W) and frame is not None:
            frame = cv2.resize(frame, (W, H), interpolation=cv2.INTER_AREA) 
        out.write(frame)
        
    out.release()
    
    
if __name__ == "__main__":
    
    srcPath = "prediction_vid.avi"
    destPath = "vid"
    
    frontFile = "0.png"
    #img = cv2.imread(frontFile)
#    for i in range(125):
#        cv2.imwrite(os.path.join(destPath, str(i)+".png"), img)
#        
#    extract_vid_frames(srcPath, destPath, i+1)
    form_video_from_frames("vid", "short_vid_254.avi")