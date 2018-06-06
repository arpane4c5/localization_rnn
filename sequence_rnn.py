#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  3 00:29:27 2018

@author: Arpan
@Description: Use RNN/LSTM on sequence of features extracted from frames.
"""

import torch
import numpy as np
import cv2
import os
import torch.nn as nn
from torch.autograd import Variable
import matplotlib.pyplot as plt
import pandas as pd
import json


import pickle

torch.manual_seed(777)  # reproducibility

# Local Paths
LABELS = "/home/hadoop/VisionWorkspace/Cricket/scripts/supporting_files/sample_set_labels/sample_labels_shots/ICC WT20"
DATASET = "/home/hadoop/VisionWorkspace/VideoData/sample_cricket/ICC WT20"

# Server Paths
if os.path.exists("/opt/datasets/cricket/ICC_WT20"):
    LABELS = "/home/arpan/VisionWorkspace/shot_detection/supporting_files/sample_set_labels/sample_labels_shots/ICC WT20"
    DATASET = "/opt/datasets/cricket/ICC_WT20"

num_classes = 5
input_size = 5  # one-hot size
hidden_size = 5  # output from the LSTM. 5 to directly predict one-hot
batch_size = 1   # one sentence
sequence_length = 6  # |ihello| == 6
num_layers = 1  # one-layer rnn


class RNN(nn.Module):

    def __init__(self, num_classes, input_size, hidden_size, num_layers):
        super(RNN, self).__init__()

        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.sequence_length = sequence_length

        self.rnn = nn.RNN(input_size=5, hidden_size=5, batch_first=True)

    def forward(self, x):
        # Initialize hidden and cell states
        # (num_layers * num_directions, batch, hidden_size) for batch_first=True
        h_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size))

        # Reshape input
        x.view(x.size(0), self.sequence_length, self.input_size)

        # Propagate input through RNN
        # Input: (batch, seq_len, input_size)
        # h_0: (num_layers * num_directions, batch, hidden_size)

        out, _ = self.rnn(x, h_0)
        return out.view(-1, num_classes)


# Split the dataset files into training, validation and test sets
def split_dataset_files():
    filenames = sorted(os.listdir(DATASET))         # read the filenames
    # filenames = [t.rsplit('.')[0] for t in filenames]   # remove the extension
    filenames = [t.split('.')[0] for t in filenames]   # remove the extension
    return filenames[:16], filenames[16:21], filenames[21:]
    
    
# function to extract the features from a list of videos
# Params: vids_lst = list of videos for which hist_diff values are to be extracted
# Return: hist_diff_all = f values of histogram diff each (256 X C) (f is no of frames)
def extract_hist_diff_vids(vids_lst, color=('g'), bins=256):
    # iterate over the videos to extract the hist_diff values
    hist_diff_all = []
    for idx, vid in enumerate(vids_lst):
        #get_hist_diff(os.path.join(DATASET, vid+'.avi'))
        diffs = getHistogramOfVideo(os.path.join(DATASET, vid+'.avi'), color, bins)
        #print "diffs : ",diffs
        print "Done : " + str(idx+1)
        hist_diff_all.append(diffs)
        # save diff_hist to disk    
        #outfile = file(os.path.join(destPath,"diff_hist.bin"), "wb")
        #np.save(outfile, diffs)
        #outfile.close()    
        #break
    return hist_diff_all

# function to get the L1 distances of histograms and plot the signal
# for getting the grayscale histogram differences, uncomment two lines
# Copied and editted from shot_detection.py script
# color=('g') for Grayscale histograms, color=('b','g','r') for RGB 
def getHistogramOfVideo(srcVideoPath, color=('g'), N=256):
    # get the VideoCapture object
    cap = cv2.VideoCapture(srcVideoPath)
    
    # if the videoCapture object is not opened then exit without traceback
    if not cap.isOpened():
        import sys
        print("Error reading the video file !!")
        sys.exit(0)

#    # create destination folder if not created already
#    if not os.path.exists(destPath):
#        os.makedirs(destPath)
    
    dimensions = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fps = cap.get(cv2.CAP_PROP_FPS)
    #fourcc = cv2.cv.CV_FOURCC(*'XVID')
    #out = cv2.VideoWriter('outputImran.avi', fourcc, fps, dimensions, True)
    #print(out)
    frameCount = 0
    #color = ('b', 'g', 'r')     # defined for 3 channels
    prev_hist = np.zeros((N, len(color)))
    curr_hist = np.zeros((N, len(color)))
    diffs = np.zeros((1, len(color)))
    while(cap.isOpened()):
        # Capture frame by frame
        ret, frame = cap.read()
        # print(ret)
    
        if ret==True:
            # frame = cv2.flip(frame)
            frameCount = frameCount + 1
            
            # Check for grayscale
            if len(color)==1:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            for i,col in enumerate(color):
                # cv2.calcHist(images, channels, mask, histSize, ranges[, hist[, accumulate]])
                curr_hist[:,i] = np.reshape(cv2.calcHist([frame], [i], None, [N], [0,N]), (N,))
            
            if frameCount > 1:
                # find the L1 distance of the current frame hist to previous frame hist 
                dist = np.sum(abs(curr_hist - prev_hist), axis=0)
                #diffs.append(dist)
                diffs = np.vstack([diffs, dist])
                #print("dist = ", type(dist), dist)           
                #print("diffs = ", type(diffs), diffs.shape)
                #waitTillEscPressed()
            np.copyto(prev_hist, curr_hist)        
            
            ### write the flipped frame
            ###out.write(frame)
            ### write frame to file
            ## Uncomment following 3 lines to get the images of the video saved to dir
            #filename = os.path.join(destPath,'f'+str(frameCount)+'.jpg')
            #cv2.imwrite(filename, frame)
            #print('Frame written', frameCount)
            #cv2.imshow('frame', frame)
            # Our operations on the frame come here
            # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Display the resulting frame
            # cv2.imshow('frame', gray)
            #if cv2.waitKey(10) == 27:
            #    print('Esc pressed')
            #    break
            
        else:
            break

    # When everything done, release the capture
    cap.release()
    return diffs


def train_model(model, destpath):
    train_files = sorted(os.listdir(DATASET))
    labels = sorted(os.listdir(LABELS))
    
    print train_files
    print labels

# function to count the number of parameters in the model
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def extract_hog_from_video(srcVid):
    seq = 0
    return seq


## Set the parameter candidates
#parameter_candidates = [
#  {'C': [1], 'kernel': ['linear']}]
# # {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
##]


# Visualize the positive and negative samples
# Params: list of numpy arrays of size nFrames-1 x Channels
def visualize_feature(samples_lst, title="Histogram", bins=300):
    
    if len(samples_lst) == 1:
        print "Cannot Visualize !! Only single numpy array in list !!"
        return
    elif len(samples_lst) > 1:
        sample = np.vstack((samples_lst[0], samples_lst[1]))
    
    # Iterate over the list to vstack those and get a single matrix
    for idx in range(2, len(samples_lst)):
        sample = np.vstack((sample, samples_lst[idx]))
        
    vals = list(sample.reshape(sample.shape[0]))
    
    plt.hist(vals, normed=True, bins=bins)
    plt.title(title)
    plt.xlabel("Bins")
    plt.ylabel("Frequency")

        
def create_dataframe(pos_samples, neg_samples):
    # create a r X 2 matrix with 2nd column of 1s (for pos_samples) or 0s (neg_samples)
    pos_feats = pos_samples[0]
    for i in range(1, len(pos_samples)):
        pos_feats = np.vstack((pos_feats, pos_samples[i]))
    pos_feats = np.hstack((pos_feats, np.ones((pos_feats.shape[0], 1))))
    
    # create similarly for negative samples
    neg_feats = neg_samples[0]
    for i in range(1, len(neg_samples)):
        neg_feats = np.vstack((neg_feats, neg_samples[i]))
    neg_feats = np.hstack((neg_feats, np.zeros((neg_feats.shape[0], 1))))
    
    # change this if no of features are different
    if pos_feats.shape[1]==2:
        cols = ["X", "Y"]
    else:
        cols = ["X1", "X2", "X3", "Y"]
    df = pd.DataFrame(np.vstack((pos_feats, neg_feats)), columns=cols)
    
    # shuffle dataframe
    df = df.sample(frac=1).reset_index(drop=True)
    return df
    
## given a dataframe in the form [X1, X2, .., Y] with Y being binary, train a model
#def train_model1(df_train):
#    #clf = svm.SVC(kernel = 'linear')
#    clf = RandomForestClassifier(max_depth=2, random_state=1234)
#    # For grid search over the parameters of gamma and kernel functions
##    clf = GridSearchCV(estimator=svm.SVC(), param_grid=parameter_candidates)
#    clf.fit(df_train.loc[:, df_train.columns != 'Y'], df_train.loc[:,'Y'])
#    
##    # Print out the results 
##    print('Best score for training data:', clf.best_score_)
##    print('Best `C`:',clf.best_estimator_.C)
##    print('Best kernel:',clf.best_estimator_.kernel)
##    print('Best `gamma`:',clf.best_estimator_.gamma)
#    
#    return clf
    
## calculate the precision, recall and f-measure for the validation of test set
## params: preds_dict: {"vid_name": [98, 138, ...], ...}
#def  calculate_accuracy(preds_dict, split = "val"):
#    # calculate metrics
#    Nt = 0      # Total no of transitions
#    Nc = 0      # No of correctly predicted transitions
#    Nd = 0      # No of deletions, not identified as cut
#    Ni = 0      # No of insertions, falsely identified as cut
#    # Iterate over the xml files (keys of preds_dict) and corresponding gt xml
#    # Calculate the metrics as defined and return the recall, precision and f-measure
#    for i,fname in enumerate(preds_dict.keys()):
#        gt_list = lab_xml.get_cuts_list_from_xml(os.path.join(LABELS, 'gt_'+fname+'.xml'))
#        test_list = preds_dict[fname]
#
#        # Calculate Nt, Nc, Nd, Ni
#        Nt = Nt + len(set(gt_list))
#        Nd = Nd + len(set(gt_list) - set(test_list))
#        Ni = Ni + len(set(test_list) - set(gt_list))
#        Nc = Nc + len(set(gt_list).intersection(set(test_list)))
#        
#        print gt_list
#        print test_list        
#        print "Nt = "+str(Nt)
#        print "Nc = "+str(Nc)
#        print "Nd = "+str(Nd)
#        print "Ni = "+str(Ni)
#        
#    # calculate the recall and precision values
#    recall = (Nc / (float)(Nc + Nd))
#    precision = (Nc / (float)(Nc + Ni))
#    f_measure = 2*precision*recall/(precision+recall)
#    return [recall, precision, f_measure]

# function to predict the cuts on the validation or test videos
def make_predictions(vids_lst, model, color, bins, split = "val"):
    # extract the hist diff features and return as a list entry for each video in vids_lst
    hist_diffs = extract_hist_diff_vids(vids_lst, color, bins)
    # form a dictionary of video names (as keys) and corresponding list of hist_diff values
    #hist_diffs_dict = dict(zip(vids_lst, hist_diffs))
    print "Extracted features !! "
    
    preds = {}
    # make predictions using the model
    for idx, vname in enumerate(vids_lst):
        # Add the additional feature
        #features = add_feature(hist_diffs[idx], )
        # make predictions using the model (returns a 1D array of 0s and 1s)
        vpreds = model.predict(hist_diffs[idx])
        # gives a n x 1 array of indices where n non-zero val occurs
        idx_preds = np.argwhere(vpreds)
        idx_preds = list(idx_preds.reshape(idx_preds.shape[0]))
        preds[vname] = idx_preds    # list of indices for positive predictions
        print(vname, idx_preds)
    
    return calculate_accuracy(preds)

# add feature: no of frames since last CUT
# sample is the numpy array containing the sequence of histogram difference values for one video
# vcuts_lst is the list of values where CUTs occur for one video
# returns sample with an additional column of counts of #Frames since last CUT
def add_feature(sample, vcuts_lst):
    # add the "#frames since last CUT" values
    nFr = []
    for i,fno in enumerate(vcuts_lst):
        if i==0:    # add values till 1st CUT
            nFr.extend(range(1,fno+1))
        else:       # reset count after each CUT and extend list
            nFr.extend(range(fno-vcuts_lst[i-1]))
    nFr.extend(range(len(sample)-len(nFr)))     # After final CUT to the end
    print "Len(nFr) : "+str(len(nFr))
    return np.hstack((sample, np.array(nFr).reshape((len(nFr),1))))


if __name__=="__main__":
    # Divide the samples files into training set, validation and test sets
    train_lst, val_lst, test_lst = split_dataset_files()
    print(train_lst, len(train_lst))
    print 60*"-"
    # specifiy for grayscale or BGR values
    color = ('g')
    bins = 256      # No of bins in the historgam
    
    # Extract the histogram difference features from the training set
    hist_diffs_train = extract_hist_diff_vids(train_lst[:1], color, bins)
    
    train_lab = [f+".json" for f in train_lst]
    val_lab = [f+".json" for f in val_lst]
    test_lab = [f+".json" for f in test_lst]
    
    # get the positions where cuts exist for training set
    tr_shots_lst = []
    for t in train_lab:
        with open(os.path.join(LABELS, t), 'r') as fobj:
            tr_shots_lst.append(json.load(fobj))
    
    pos_samples = []
    neg_samples = []
    
    #idx2char = ['h', 'i', 'e', 'l', 'o']
    
    x_data = hist_diffs_train[0]    # 4404 x 1
    
    # seq len = 4404 (len of video)
    # dim of vector = 1 (more for HOG etc.)
    # Batch = 1
    
    # Teach hihell -> ihello
    #x_data = [[0, 1, 0, 2, 3, 3]]   # hihell
    #x_one_hot = [[[1, 0, 0, 0, 0],   # h 0
    #              [0, 1, 0, 0, 0],   # i 1
    #              [1, 0, 0, 0, 0],   # h 0
    #              [0, 0, 1, 0, 0],   # e 2
    #              [0, 0, 0, 1, 0],   # l 3
    #              [0, 0, 0, 1, 0]]]  # l 3
    
    #y_data = [1, 0, 2, 3, 3, 4]    # ihello
    
    # As we have one batch of samples, we will change them to variables only once
    # Convert to shape (1, 4404, 1)  (Feature Vector size is 1)
    inputs = Variable(torch.Tensor(np.expand_dims(x_data, 0)))
    labels = Variable(torch.LongTensor(y_data))
    
    # get vector sequences for video frames
    # sequences in hist_diffs_train 
    
    # Prepare the RNN 

#    for idx, sample in enumerate(hist_diffs_train):
#        #sample = add_feature(sample, cuts_lst[idx])     # add feature #Frames since last CUT
#        pos_samples.append(sample[cuts_lst[idx],:])     # append a np array of pos samples
#        neg_indices = list(set(range(len(sample))) - set(cuts_lst[idx]))
#        # subset
#        neg_samples.append(sample[neg_indices,:])
#    
#    
#    # Save the pos_samples and neg_samples lists to disk
##    with open("pos_samples_bgr.pkl", "wb") as fp:
##        pickle.dump(pos_samples, fp)
##        
##    with open("neg_samples_bgr.pkl", "wb") as fp:
##        pickle.dump(neg_samples, fp)
#        
#    # Read the lists from disk to the pickle files
##    with open("pos_samples.pkl", "rb") as fp:
##        pos_samples = pickle.load(fp)
##    
##    with open("neg_samples.pkl", "rb") as fp:
##        neg_samples = pickle.load(fp)
#        
#    #print "Visualizing positive and negative training samples ..."
#    #visualize_feature(pos_samples, "Positive Samples", 30)
#    #visualize_feature(neg_samples, "Negative Samples", 300)
#    
#    df = create_dataframe(pos_samples, neg_samples)
#    print df.shape
#    
#    # Training a model given a dataframe
#    print "Training model ..."
#    trained_model = train_model1(df)
#    
#    # get predictions on the validation or test set videos
#    #pr = svm_model.predict(df.sample(frac=0.001).loc[:,['X']])
#    
#    # extract the validation/test set features and make predictions on the same
#    print "Predicting on the validation set !!"
#    [recall, precision, f_measure] = make_predictions(val_lst, trained_model, color, bins)
#    print "Precision : "+str(precision)
#    print "Recall : "+ str(recall)
#    print "F-measure : "+str(f_measure)
#    
#    print "Predicting on the test set !!"
#    [recall, precision, f_measure] = make_predictions(test_lst, trained_model, color, bins)
#    print "Precision : "+str(precision)
#    print "Recall : "+ str(recall)
#    print "F-measure : "+str(f_measure)
#    
#    ## Save model to disk
#    from sklearn.externals import joblib
#    joblib.dump(trained_model, "sbd_model_RF_histDiffs_gray.pkl")
#    
#    #######################################################
#    # Extend 1:
#    # Add feature: #Frames till the last shot boundary: Will it be correct feature
#    # How to handle testing set feature. A single false +ve will screw up the subsequent 
#    # predictions.
#    
#    #######################################################
#    
#    # Extend 2: Experiment with Random Forests, decision trees and Bayesian Inf
#    #
#    
#    #######################################################
#    
#    # Extend 3 : Learn a CNN architecture
#    
#    #######################################################
#    
#    # Extend 4 : Learn an RNN by extracting features 
#
#    #extract_features()
#    # create a model 
#    #import model_c3d as c3d
#    
#    #model = c3d.C3D()
#    
#    #print model
#    #train_model(model, "data")
#    
#    # count no. of parameters in the model
#    #model_parameters = filter(lambda p: p.requires_grad, model.parameters())
#    #params = sum([np.prod(p.size()) for p in model_parameters])
#    # or call count_paramters(model)  
#    #print "#Parameters : {} ".format(count_parameters(model))
#    
#    # Creation of a training set, validation set, test set meta info file.
#    
#    # Train the model
#    
#    # 
#    
#    # Instantiate RNN model
#    rnn = RNN(num_classes, input_size, hidden_size, num_layers)
#    print(rnn)
#    
#    # Set loss and optimizer function
#    # CrossEntropyLoss = LogSoftmax + NLLLoss
#    criterion = torch.nn.CrossEntropyLoss()
#    optimizer = torch.optim.Adam(rnn.parameters(), lr=0.1)
#    
#    # Train the model
#    for epoch in range(100):
#        outputs = rnn(inputs)
#        optimizer.zero_grad()
#        loss = criterion(outputs, labels)
#        loss.backward()
#        optimizer.step()
#        _, idx = outputs.max(1)
#        idx = idx.data.numpy()
#        result_str = [idx2char[c] for c in idx.squeeze()]
#        print("epoch: %d, loss: %1.3f" % (epoch + 1, loss.data[0]))
#        print("Predicted string: ", ''.join(result_str))
#    
#    print("Learning finished!")    
    