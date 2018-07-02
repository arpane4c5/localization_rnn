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
from torch.utils.data import DataLoader
#import matplotlib.pyplot as plt
import pandas as pd
import time
import shutil
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from Video_Dataset import VideoDataset

torch.manual_seed(777)  # reproducibility

# Local Paths
LABELS = "/home/hadoop/VisionWorkspace/Cricket/scripts/supporting_files/sample_set_labels/sample_labels_shots/ICC WT20"
DATASET = "/home/hadoop/VisionWorkspace/VideoData/sample_cricket/ICC WT20"

# Server Paths
if os.path.exists("/opt/datasets/cricket/ICC_WT20"):
    LABELS = "/home/arpan/VisionWorkspace/shot_detection/supporting_files/sample_set_labels/sample_labels_shots/ICC WT20"
    DATASET = "/opt/datasets/cricket/ICC_WT20"

num_classes = 2
input_size = 1  # one-hot size
hidden_size = 1  # output from the LSTM. 5 to directly predict one-hot
batch_size = 1   # one sentence
sequence_length = 4404  # |ihello| == 6
num_layers = 1  # one-layer rnn
THRESHOLD = 0.5

class RNNClassifier(nn.Module):
    # Our model

    def __init__(self, input_size, hidden_size, output_size, n_layers=1, bidirectional=False):
        super(RNNClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.n_directions = int(bidirectional) + 1

        #self.embedding = nn.Embedding(input_size, hidden_size)
        #self.rnn = nn.RNN(input_size=input_size,
        #                  hidden_size=hidden_size, batch_first=True)
        self.gru = nn.GRU(input_size, hidden_size, n_layers, batch_first=True,
                          bidirectional=bidirectional)
        self.fc = nn.Linear(hidden_size, output_size)
        #self.soft = nn.Softmax()

    def forward(self, input):
        # Note: we run this all at once (over the whole input sequence)
        # input shape: B x S (input size)
        # transpose to make S(sequence) x B (batch)
        #input = input.t()
        #batch_size = input.size(1)
        batch_size = input.size(0)
        seq_len = input.size(1)        
        #print "Seq Len : {} :: Batch size : {}".format(seq_len, batch_size)

        # Make a hidden
        hidden = self._init_hidden(batch_size)

        # Embedding S x B -> S x B x I (embedding size)
        #embedded = self.embedding(input.view(seq_len, batch_size, -1))
        # B X S X 1152
        embedded = input

        # Pack them up nicely
        #gru_input = pack_padded_sequence(
        #    embedded, seq_lengths.data.cpu().numpy())

        # To compact weights again call flatten_parameters().
        self.gru.flatten_parameters()
        #output, hidden = self.rnn(embedded, hidden)
        output, hidden = self.gru(embedded, hidden)

        # Use the last layer output as FC's input
        # No need to unpack, since we are going to use hidden
        return self.fc(output.contiguous().view(-1, self.hidden_size))
        #fc_output = self.fc(hidden[-1])
        #return fc_output

    def _init_hidden(self, batch_size):
         #* self.n_directions
        hidden = torch.zeros(self.n_layers * self.n_directions, batch_size, \
                             self.hidden_size)
        return create_variable(hidden)


#class RNN(nn.Module):
#
#    def __init__(self, num_classes, input_size, hidden_size, num_layers):
#        super(RNN, self).__init__()
#
#        self.num_classes = num_classes
#        self.num_layers = num_layers
#        self.input_size = input_size
#        self.hidden_size = hidden_size
#        self.sequence_length = sequence_length
#
#        self.rnn = nn.RNN(input_size=self.input_size, hidden_size=self.hidden_size, \
#                          batch_first=True)
#
#    def forward(self, x):
#        # Initialize hidden and cell states
#        # (num_layers * num_directions, batch, hidden_size) for batch_first=True
#        h_0 = Variable(torch.zeros(
#            self.num_layers, x.size(0), self.hidden_size))
#
#        # Reshape input
#        x.view(x.size(0), self.sequence_length, self.input_size)
#
#        # Propagate input through RNN
#        # Input: (batch, seq_len, input_size)
#        # h_0: (num_layers * num_directions, batch, hidden_size)
#
#        out, _ = self.rnn(x, h_0)
#        return out.view(-1, num_classes-1)


def create_variable(tensor):
    # Do cuda() before wrapping with variable
#    if torch.cuda.is_available():
#        return Variable(tensor.cuda())
#    else:
    return Variable(tensor)

# Split the dataset files into training, validation and test sets
def split_dataset_files():
    filenames = sorted(os.listdir(DATASET))         # read the filenames
    # filenames = [t.rsplit('.')[0] for t in filenames]   # remove the extension
    filenames = [t.split('.')[0] for t in filenames]   # remove the extension
    return filenames[:16], filenames[16:21], filenames[21:]
    
# function to extract the features from a list of videos
# Params: vids_lst = list of videos for which hist_diff values are to be extracted
# Return: hist_diff_all = f values of histogram diff each (256 X C) (f is no of frames)
#def extract_hist_diff_vids(vids_lst, color=('g'), bins=256):
#    # iterate over the videos to extract the hist_diff values
#    hist_diff_all = []
#    for idx, vid in enumerate(vids_lst):
#        #get_hist_diff(os.path.join(DATASET, vid+'.avi'))
#        diffs = getHistogramOfVideo(os.path.join(DATASET, vid+'.avi'), color, bins)
#        #print "diffs : ",diffs
#        print "Done : " + str(idx+1)
#        hist_diff_all.append(diffs)
#        # save diff_hist to disk    
#        #outfile = file(os.path.join(destPath,"diff_hist.bin"), "wb")
#        #np.save(outfile, diffs)
#        #outfile.close()    
#        #break
#    return hist_diff_all

# function to get the L1 distances of histograms and plot the signal
# for getting the grayscale histogram differences, uncomment two lines
# Copied and editted from shot_detection.py script
# color=('g') for Grayscale histograms, color=('b','g','r') for RGB 
#def getHistogramOfVideo(srcVideoPath, color=('g'), N=256):
#    # get the VideoCapture object
#    cap = cv2.VideoCapture(srcVideoPath)
#    
#    # if the videoCapture object is not opened then exit without traceback
#    if not cap.isOpened():
#        import sys
#        print("Error reading the video file !!")
#        sys.exit(0)
#
##    # create destination folder if not created already
##    if not os.path.exists(destPath):
##        os.makedirs(destPath)
#    
#    dimensions = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
#    fps = cap.get(cv2.CAP_PROP_FPS)
#    #fourcc = cv2.cv.CV_FOURCC(*'XVID')
#    #out = cv2.VideoWriter('outputImran.avi', fourcc, fps, dimensions, True)
#    #print(out)
#    frameCount = 0
#    #color = ('b', 'g', 'r')     # defined for 3 channels
#    prev_hist = np.zeros((N, len(color)))
#    curr_hist = np.zeros((N, len(color)))
#    diffs = np.zeros((1, len(color)))
#    while(cap.isOpened()):
#        # Capture frame by frame
#        ret, frame = cap.read()
#        # print(ret)
#    
#        if ret==True:
#            # frame = cv2.flip(frame)
#            frameCount = frameCount + 1
#            
#            # Check for grayscale
#            if len(color)==1:
#                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#            
#            for i,col in enumerate(color):
#                # cv2.calcHist(images, channels, mask, histSize, ranges[, hist[, accumulate]])
#                curr_hist[:,i] = np.reshape(cv2.calcHist([frame], [i], None, [N], [0,N]), (N,))
#            
#            if frameCount > 1:
#                # find the L1 distance of the current frame hist to previous frame hist 
#                dist = np.sum(abs(curr_hist - prev_hist), axis=0)
#                #diffs.append(dist)
#                diffs = np.vstack([diffs, dist])
#                #print("dist = ", type(dist), dist)           
#                #print("diffs = ", type(diffs), diffs.shape)
#                #waitTillEscPressed()
#            np.copyto(prev_hist, curr_hist)        
#            
#            ### write the flipped frame
#            ###out.write(frame)
#            ### write frame to file
#            ## Uncomment following 3 lines to get the images of the video saved to dir
#            #filename = os.path.join(destPath,'f'+str(frameCount)+'.jpg')
#            #cv2.imwrite(filename, frame)
#            #print('Frame written', frameCount)
#            #cv2.imshow('frame', frame)
#            # Our operations on the frame come here
#            # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#            
#            # Display the resulting frame
#            # cv2.imshow('frame', gray)
#            #if cv2.waitKey(10) == 27:
#            #    print('Esc pressed')
#            #    break
#            
#        else:
#            break
#
#    # When everything done, release the capture
#    cap.release()
#    return diffs


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
    
    #return calculate_accuracy(preds)
    return preds

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


def getNFrames(vid):
    cap = cv2.VideoCapture(vid)
    if not cap.isOpened():
        import sys
        print "Capture Object not opened ! Abort"
        sys.exit(0)
        
    l = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return l

# Iteratively take the batch information and extract the feature sequences from the videos
# datasetpath : Prefix of the path to the dataset containing the videos
# videoFiles : list/tuple of filenames for the videos (size n)
# sequences :  list of start frame numbers and end frame numbers 
# sequences[0] and [1] are torch.LongTensor of size n each.
# returns a list of lists. Inner list contains a sequence of arrays 
def getFeatureVectors(datasetpath, videoFiles, sequences):
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
    Load the features of the train/val/test set into a dictionary
    Dictionary has key as the filename of video and value as the numpy feature 
    matrix
    """
    feats = {}
    for k in keys:
        featpath = os.path.join(OFfeaturesPath, k.rsplit('.', 1)[0])+".bin"
        with open(featpath, "rb") as fobj:
            feats[k] = np.load(fobj)
            
    print "Features loaded into dictionary ..."
    return feats
        
    

def getFeatureVectorsFromDump(OFfeatures, videoFiles, sequences):
    """Pass a dictionary of features {vidname: numpy matrix, ...}, videoFiles
    is the list of filenames for a batch, sequences is the start and end frame numbers
    in the batch videos to be sampled.
    """
    #grid_size = 20
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


## Train cycle
#def train(train_loader):
#    total_loss = 0
#
#    for i, (names, countries) in enumerate(train_loader, 1):
#        input, seq_lengths, target = make_variables(names, countries)
#        output = classifier(input, seq_lengths)
#
#        loss = criterion(output, target)
#        total_loss += loss.data[0]
#
#        classifier.zero_grad()
#        loss.backward()
#        optimizer.step()
#
#        if i % 10 == 0:
#            print('[{}] Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.2f}'.format(
#                time_since(start), epoch,  i *
#                len(names), len(train_loader.dataset),
#                100. * i * len(names) / len(train_loader.dataset),
#                total_loss / i * len(names)))
#
#    return total_loss


## Testing cycle
#def test(name=None):
#    # Predict for a given name
#    if name:
#        input, seq_lengths, target = make_variables([name], [])
#        output = classifier(input, seq_lengths)
#        pred = output.data.max(1, keepdim=True)[1]
#        country_id = pred.cpu().numpy()[0][0]
#        print(name, "is", train_dataset.get_country(country_id))
#        return
#
#    print("evaluating trained model ...")
#    correct = 0
#    train_data_size = len(test_loader.dataset)
#
#    for names, countries in test_loader:
#        input, seq_lengths, target = make_variables(names, countries)
#        output = classifier(input, seq_lengths)
#        pred = output.data.max(1, keepdim=True)[1]
#        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
#
#    print('\nTest set: Accuracy: {}/{} ({:.0f}%)\n'.format(
#        correct, train_data_size, 100. * correct / train_data_size))


# Inputs: feats: list of lists
def make_variables(feats, labels):
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
            target.extend([0]*(labels[0][i]-1) + [1]*labels[1][i])
        else:
            target.extend([0]*labels[0][i] + [1]*(labels[1][i]-1))
    # Form a wrap into a tensor variable as B X S X I
    return create_variable(feats), create_variable(torch.Tensor(target))

def save_checkpoint(state, is_best, save_path, filename):
    filename = os.path.join(save_path, filename)
    torch.save(state, filename)
    if is_best:
        bestname = os.path.join(save_path, 'model_best.pth.tar')
        shutil.copyfile(filename, bestname)
        

if __name__=="__main__":
    # Divide the samples files into training set, validation and test sets
    train_lst, val_lst, test_lst = split_dataset_files()
    print(train_lst, len(train_lst))
    print 60*"-"
    # specifiy for grayscale or BGR values
    color = ('g')
    bins = 256      # No of bins in the historgam
    gridSize = 20
    
    # Extract the histogram difference features from the training set
    #hist_diffs_train = extract_hist_diff_vids(train_lst[:1], color, bins)
    
    train_lab = [f+".json" for f in train_lst]
    val_lab = [f+".json" for f in val_lst]
    test_lab = [f+".json" for f in test_lst]
    
    #####################################################################
    
    tr_labs = [os.path.join(LABELS, f) for f in train_lab]
    sizes = [getNFrames(os.path.join(DATASET, f+".avi")) for f in train_lst]
    print "Size : {}".format(sizes)
    hlDataset = VideoDataset(tr_labs, sizes, is_train_set = True)
    print hlDataset.__len__()
    
    #####################################################################
    # Run extract_denseOF_par.py before executing this file, using same grid_size
    # Features already extracted and dumped to disk 
    # Read those features using the given path and grid size
    OFfeaturesPath = os.path.join(os.getcwd(),"OF_grid"+str(gridSize))
    
    # Uncomment the lines below to extract features for a different gridSize
#    from extract_denseOF_par import extract_dense_OF_vids
#    start = time.time()
#    extract_dense_OF_vids(DATASET, OFfeaturesPath, grid_size=gridSize, stop='all')
#    end = time.time()
#    print "Total execution time : "+str(end-start)
    
    #####################################################################
    
    
    # Parameters and DataLoaders
    HIDDEN_SIZE = 100
    N_LAYERS = 1
    BATCH_SIZE = 20
    N_EPOCHS = 2
    N_CHARS = 1152      # taking grid_size = 20 get this feature vector size 
    
    #test_dataset = NameDataset(is_train_set=False)
    #test_loader = DataLoader(dataset=test_dataset,
    #                         batch_size=BATCH_SIZE, shuffle=True)
    
    #N_COUNTRIES = len(train_dataset.get_countries())
        
    classifier = RNNClassifier(N_CHARS, HIDDEN_SIZE, 1, N_LAYERS)
#    if torch.cuda.device_count() > 1:
#        print("Let's use", torch.cuda.device_count(), "GPUs!")
#        # dim = 0 [33, xxx] -> [11, ...], [11, ...], [11, ...] on 3 GPUs
#        classifier = nn.DataParallel(classifier)
#
#    if torch.cuda.is_available():
#        classifier.cuda()

    optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)
    #criterion = nn.CrossEntropyLoss()
    sigm = nn.Sigmoid()
    criterion = nn.BCELoss()

    start = time.time()
    print("Training for %d epochs..." % N_EPOCHS)
    
    # Create a DataLoader object and sample batches of examples. 
    # These batch samples are used to extract the features from videos parallely
    train_loader = DataLoader(dataset=hlDataset, batch_size=BATCH_SIZE, shuffle=True)

    print(len(train_loader.dataset))
    for epoch in range(10):
        total_loss = 0
        for i, (keys, seqs, labels) in enumerate(train_loader):
            # Run your training process
            #print(epoch, i) #, "keys", keys, "Sequences", seqs, "Labels", labels)
            #feats = getFeatureVectors(DATASET, keys, seqs)      # Parallelize this
            feats = getFeatureVectorsFromDump(OFfeaturesPath, keys, seqs)
            #break

            # Training starts here
            inputs, target = make_variables(feats, labels)
            output = classifier(inputs)

            loss = criterion(sigm(output.view(output.size(0))), target)
            total_loss += loss.data[0]

            classifier.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 2 == 0:
                print('Train Epoch: {} :: Loss: {:.2f}'.format(epoch, total_loss))
            if (i+1) % 10 == 0:
                break
    
    
    # Save only the model params
    torch.save(classifier.state_dict(), "gru_100_epoch10_BCE_temp.pt")
    
    # To load the params into model
    #the_model = RNNClassifier(N_CHARS, HIDDEN_SIZE, 1, N_LAYERS)
    #the_model.load_state_dict(torch.load("gru_100_epoch10_BCE.pt"))
    
    classifier.load_state_dict(torch.load("gru_100_epoch10_BCE_temp.pt"))
    
#    save_checkpoint({
#            'epoch': epoch + 1,
#            'arch': args.arch,
#            'state_dict': classifier.state_dict(),
#            'best_prec1': best_prec1,
#            'optimizer' : optimizer.state_dict(),
#        }, is_best)
#    #####################################################################
#    
#    # Loading and resuming from dictionary
#    # Refer : https://github.com/pytorch/examples/blob/master/imagenet/main.py#L139
#    if args.resume:
#        if os.path.isfile(args.resume):
#            print("=> loading checkpoint '{}'".format(args.resume))
#            checkpoint = torch.load(args.resume)
#            args.start_epoch = checkpoint['epoch']
#            best_prec1 = checkpoint['best_prec1']
#            model.load_state_dict(checkpoint['state_dict'])
#            optimizer.load_state_dict(checkpoint['optimizer'])
#            print("=> loaded checkpoint '{}' (epoch {})"
#                  .format(args.resume, checkpoint['epoch']))
#        else:
#            print("=> no checkpoint found at '{}'".format(args.resume))
            
    #####################################################################
    
    # Test a video or calculate the accuracy using the learned model
    val_labs = [os.path.join(LABELS, f) for f in val_lab]
    val_sizes = [getNFrames(os.path.join(DATASET, f+".avi")) for f in val_lst]
    print "Size : {}".format(val_sizes)
    hlvalDataset = VideoDataset(val_labs, val_sizes, is_train_set = False)
    print hlvalDataset.__len__()
    
    # Create a DataLoader object and sample batches of examples. 
    # These batch samples are used to extract the features from videos parallely
    val_loader = DataLoader(dataset=hlvalDataset, batch_size=BATCH_SIZE, shuffle=False)
    print(len(val_loader.dataset))
    correct = 0
    val_keys = []
    predictions = []
    
    for i, (keys, seqs, labels) in enumerate(val_loader):
        
        # Testing on the sample
        feats = getFeatureVectors(DATASET, keys, seqs)      # Parallelize this
        #break
        # Validation stage
        inputs, target = make_variables(feats, labels)
        output = classifier(inputs) # of size (BATCHESxSeqLen) X 1

        #pred = output.data.max(1, keepdim=True)[1]  # get max value in each row
        pred_probs = sigm(output.view(output.size(0))).data  # get the normalized values (0-1)
        #preds = pred_probs > THRESHOLD  # ByteTensor
        #correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        val_keys.append(keys)
        predictions.append(pred_probs)  # append the 
        
        
        #loss = criterion(m(output.view(output.size(0))), target)
        #total_loss += loss.data[0]

        if i % 2 == 0:
            print('i: {} :: Val keys: {} : seqs : {}'.format(i, keys, seqs)) #keys, pred_probs))
        #if (i+1) % 10 == 0:
        #    break
    
#    
#    for names, countries in test_loader:
#        input, seq_lengths, target = make_variables(names, countries)
#        output = classifier(input, seq_lengths)
#        pred = output.data.max(1, keepdim=True)[1]
        
    #####################################################################
    
    
    
#    
#    # Set loss and optimizer function
#    # CrossEntropyLoss = LogSoftmax + NLLLoss
#    criterion = torch.nn.CrossEntropyLoss()
#    optimizer = torch.optim.Adam(rnn.parameters(), lr=0.1)
#    
#    # Train the model
#    for epoch in range(10):
#        outputs = rnn(inputs)
#        optimizer.zero_grad()
#        loss = criterion(outputs, labels)
#        loss.backward()
#        optimizer.step()
#        _, idx = outputs.max(1)
#        idx = idx.data.numpy()
#        #result_str = [idx2char[c] for c in idx.squeeze()]
#        
#        print("epoch: %d, loss: %1.3f" % (epoch + 1, loss.data[0]))
#        #print("Predicted string: ", ''.join(idx))
#    
#    print("Learning finished!")    

    #####################################################################
    
#    # get the positions where cuts exist for training set
#    tr_shots_lst = []
#    for t in train_lab:
#        with open(os.path.join(LABELS, t), 'r') as fobj:
#            tr_shots_lst.append(json.load(fobj))
#    
#    pos_samples = []
#    neg_samples = []
#    
#    #idx2char = ['h', 'i', 'e', 'l', 'o']
#    
#    x_data = hist_diffs_train[0]    # 4404 x 1
#    
#    for idx, v in enumerate(tr_shots_lst):  # v is a dict with labels
#        vid_pos = v['ICC WT20/'+train_lab[idx].rsplit('.', 1)[0]+'.avi']        
#        y_data = get_vid_labels_vec(vid_pos, hist_diffs_train[idx].shape[0])
#        break
#                            
#    y_data.append(0)
#    y_data = y_data[1:]
#    # seq len = 4404 (len of video)
#    # dim of vector = 1 (more for HOG etc.)
#    # Batch = 1

#    
#    # As we have one batch of samples, we will change them to variables only once
#    # Convert to shape (1, 4404, 1)  (Feature Vector size is 1)
#    inputs = Variable(torch.Tensor(np.expand_dims(x_data, 0)))
#    labels = Variable(torch.LongTensor(y_data))



#    for idx, sample in enumerate(hist_diffs_train):
#        #sample = add_feature(sample, cuts_lst[idx])     # add feature #Frames since last CUT
#        pos_samples.append(sample[cuts_lst[idx],:])     # append a np array of pos samples
#        neg_indices = list(set(range(len(sample))) - set(cuts_lst[idx]))
#        # subset
#        neg_samples.append(sample[neg_indices,:])
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