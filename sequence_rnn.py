#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  3 00:29:27 2018

@author: Arpan
@Description: Use RNN/LSTM on sequence of features extracted from frames for action 
localization of cricket strokes on Highlight videos dataset.
"""

import torch
import numpy as np
import cv2
import os
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
#import matplotlib.pyplot as plt
import pickle
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

THRESHOLD = 0.5
# Parameters and DataLoaders
HIDDEN_SIZE = 1000
N_LAYERS = 1
BATCH_SIZE = 256
N_EPOCHS = 100
INP_VEC_SIZE = 1152      # taking grid_size = 20 get this feature vector size 
#INP_VEC_SIZE = 576     # only magnitude features
SEQ_SIZE = 10

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
        #self.gru1 = nn.GRU(hidden_size, hidden_size, n_layers, batch_first=True,
        #                  bidirectional=bidirectional)
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
        #self.gru1.flatten_parameters()
        #output, hidden = self.gru1(output, hidden)

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
    if torch.cuda.is_available():
        return Variable(tensor.cuda())
    else:
        return Variable(tensor)

# Split the dataset files into training, validation and test sets
def split_dataset_files():
    filenames = sorted(os.listdir(DATASET))         # read the filenames
    # filenames = [t.rsplit('.')[0] for t in filenames]   # remove the extension
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
     Iteratively take the batch information and extract the feature sequences from the videos
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

if __name__=="__main__":
    # Divide the samples files into training set, validation and test sets
    train_lst, val_lst, test_lst = split_dataset_files()
    print(train_lst, len(train_lst))
    print 60*"-"
    # specifiy for grayscale or BGR values
    color = ('g')
    bins = 256      # No of bins in the historgam
    gridSize = 20
    
    train_lab = [f+".json" for f in train_lst]
    val_lab = [f+".json" for f in val_lst]
    test_lab = [f+".json" for f in test_lst]
    
    #####################################################################
    
    tr_labs = [os.path.join(LABELS, f) for f in train_lab]
    sizes = [getNFrames(os.path.join(DATASET, f+".avi")) for f in train_lst]
    print "Size : {}".format(sizes)
    hlDataset = VideoDataset(tr_labs, sizes, seq_size=SEQ_SIZE, is_train_set = True)
    print hlDataset.__len__()
    
    #####################################################################
    # Run extract_denseOF_par.py before executing this file, using same grid_size
    # Features already extracted and dumped to disk 
    # Read those features using the given path and grid size
    #OFfeaturesPath = os.path.join(os.getcwd(),"OF_grid"+str(gridSize))
    HOGfeaturesPath = os.path.join(os.getcwd(),"hog_feats_new")
    
    # Uncomment the lines below to extract features for a different gridSize
#    from extract_denseOF_par import extract_dense_OF_vids
#    start = time.time()
#    extract_dense_OF_vids(DATASET, OFfeaturesPath, grid_size=gridSize, stop='all')
#    end = time.time()
#    print "Total execution time : "+str(end-start)
    
    #####################################################################
    
    # Create a DataLoader object and sample batches of examples. 
    # These batch samples are used to extract the features from videos parallely
    train_loader = DataLoader(dataset=hlDataset, batch_size=BATCH_SIZE, shuffle=True)

    train_losses = []
    # read into dictionary {vidname: np array, ...}
    print("Loading features from disk...")
    #OFfeatures = readAllOFfeatures(OFfeaturesPath, train_lst)
    HOGfeatures = readAllHOGfeatures(HOGfeaturesPath, train_lst)
    print(len(train_loader.dataset))
    
    INP_VEC_SIZE = len(HOGfeatures[HOGfeatures.keys()[0]][0])   # list of vals
    
    # Creating the RNN and training
    classifier = RNNClassifier(INP_VEC_SIZE, HIDDEN_SIZE, 1, N_LAYERS)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [33, xxx] -> [11, ...], [11, ...], [11, ...] on 3 GPUs
        classifier = nn.DataParallel(classifier)

    if torch.cuda.is_available():
        classifier.cuda()

    optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)
    #criterion = nn.CrossEntropyLoss()
    sigm = nn.Sigmoid()
    criterion = nn.BCELoss()

    start = time.time()
    
    print("Training for %d epochs..." % N_EPOCHS)
    for epoch in range(N_EPOCHS):
        total_loss = 0
        for i, (keys, seqs, labels) in enumerate(train_loader):
            # Run your training process
            #print(epoch, i) #, "keys", keys, "Sequences", seqs, "Labels", labels)
            #feats = getFeatureVectors(DATASET, keys, seqs)   # Takes time. Do not use
            #batchFeats = getFeatureVectorsFromDump(OFfeatures, keys, seqs, motion=True)
            batchFeats = getFeatureVectorsFromDump(HOGfeatures, keys, seqs, motion=False)
            #break

            # Training starts here
            #inputs, target = make_variables(batchFeats, labels, motion=True)
            inputs, target = make_variables(batchFeats, labels, motion=False)
            output = classifier(inputs)

            loss = criterion(sigm(output.view(output.size(0))), target)
            total_loss += loss.data[0]

            classifier.zero_grad()
            loss.backward()
            optimizer.step()

            #if i % 2 == 0:
            #    print('Train Epoch: {} :: Loss: {:.2f}'.format(epoch, total_loss))
            #if (i+1) % 10 == 0:
            #    break
        train_losses.append(total_loss)
        print('Train Epoch: {} :: Loss: {:.2f}'.format(epoch+1, total_loss))
    
    
    # Save only the model params
    torch.save(classifier.state_dict(), "gru_100_epoch10_BCE.pt")
    print "Model saved to disk..."
    
    # Save losses to a txt file
    with open("losses.pkl", "w") as fp:
        pickle.dump(train_losses, fp)
    
    # To load the params into model
    ##the_model = RNNClassifier(INP_VEC_SIZE, HIDDEN_SIZE, 1, N_LAYERS)
    ##the_model.load_state_dict(torch.load("gru_100_epoch10_BCE.pt"))    
    #classifier.load_state_dict(torch.load("gru_100_epoch10_BCE.pt"))
    
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
    print "Prediction video meta info."
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
    print "Loading validation/test features from disk..."
    #OFValFeatures = readAllOFfeatures(OFfeaturesPath, val_lst)
    HOGValFeatures = readAllHOGfeatures(HOGfeaturesPath, val_lst)   
    print("Predicting on the validation/test videos...")
    for i, (keys, seqs, labels) in enumerate(val_loader):
        
        # Testing on the sample
        #feats = getFeatureVectors(DATASET, keys, seqs)      # Parallelize this
        #batchFeats = getFeatureVectorsFromDump(OFValFeatures, keys, seqs, motion=True)
        batchFeats = getFeatureVectorsFromDump(HOGValFeatures, keys, seqs, motion=False)
        #break
        # Validation stage
        #inputs, target = make_variables(batchFeats, labels, motion=True)
        inputs, target = make_variables(batchFeats, labels, motion=False)
        output = classifier(inputs) # of size (BATCHESxSeqLen) X 1

        #pred = output.data.max(1, keepdim=True)[1]  # get max value in each row
        pred_probs = sigm(output.view(output.size(0))).data  # get the normalized values (0-1)
        #preds = pred_probs > THRESHOLD  # ByteTensor
        #correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        val_keys.append(keys)
        predictions.append(pred_probs)  # append the 
        
        #loss = criterion(m(output.view(output.size(0))), target)
        #total_loss += loss.data[0]

        #if i % 2 == 0:
        #    print('i: {} :: Val keys: {} : seqs : {}'.format(i, keys, seqs)) #keys, pred_probs))
        #if (i+1) % 10 == 0:
        #    break
    print "Predictions done on validation/test set..."
    #####################################################################
    
    with open("predictions.pkl", "wb") as fp:
        pickle.dump(predictions, fp)
    
    with open("val_keys.pkl", "wb") as fp:
        pickle.dump(val_keys, fp)
    
#    with open("predictions.pkl", "rb") as fp:
#        predictions = pickle.load(fp)
#    
#    with open("val_keys.pkl", "rb") as fp:
#        val_keys = pickle.load(fp)
    
    from get_localizations import getLocalizations
    from get_localizations import getVidLocalizations
    threshold = 0.5
    seq_threshold = 0.5
    # [4949, 4369, 4455, 4317, 4452]
    localization_dict = getLocalizations(val_keys, predictions, BATCH_SIZE, \
                                         threshold, seq_threshold)

    print localization_dict
    
    import json        
#    for i in range(0,101,10):
#        filtered_shots = filter_action_segments(localization_dict, epsilon=i)
#        filt_shots_filename = "predicted_localizations_th0_5_filt"+str(i)+".json"
#        with open(filt_shots_filename, 'w') as fp:
#            json.dump(filtered_shots, fp)

    # Apply filtering    
    i = 60  # optimum
    filtered_shots = filter_action_segments(localization_dict, epsilon=i)
    #i = 7  # optimum
    #filtered_shots = filter_non_action_segments(filtered_shots, epsilon=i)
    filt_shots_filename = "predicted_localizations_th0_5_filt"+str(i)+".json"
    with open(filt_shots_filename, 'w') as fp:
        json.dump(filtered_shots, fp)
    print("Prediction file written to disk !!")
    #####################################################################
