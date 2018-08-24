#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  3 00:29:27 2018

@author: Arpan
@Description: Use RNN/LSTM on sequence of features extracted from frames for action 
localization of cricket strokes on Highlight videos dataset.
"""

import torch
import os
import torch.nn as nn
from torch.utils.data import DataLoader
#import matplotlib.pyplot as plt
import pickle
import time
import utils
from torch.autograd import Variable
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
SEQ_SIZE = 2
threshold = 0.5
seq_threshold = 0.5

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
        return utils.create_variable(hidden)


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


if __name__=="__main__":
    # Divide the samples files into training set, validation and test sets
    train_lst, val_lst, test_lst = utils.split_dataset_files(DATASET)
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
    sizes = [utils.getNFrames(os.path.join(DATASET, f+".avi")) for f in train_lst]
    print "Size : {}".format(sizes)
    hlDataset = VideoDataset(tr_labs, sizes, seq_size=SEQ_SIZE, is_train_set = True)
    print hlDataset.__len__()
    
    #####################################################################
    # Run extract_denseOF_par.py before executing this file, using same grid_size
    # Features already extracted and dumped to disk 
    # Read those features using the given path and grid size
    OFfeaturesPath = os.path.join(os.getcwd(),"OF_grid"+str(gridSize))
    #HOGfeaturesPath = os.path.join(os.getcwd(),"hog_feats_new")
    
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
    OFfeatures = utils.readAllOFfeatures(OFfeaturesPath, train_lst)
    #HOGfeatures = utils.readAllHOGfeatures(HOGfeaturesPath, train_lst)
    print(len(train_loader.dataset))
    
    INP_VEC_SIZE = len(OFfeatures[OFfeatures.keys()[0]][0])   # list of vals
    #INP_VEC_SIZE = len(HOGfeatures[HOGfeatures.keys()[0]][0])   # list of vals
    print("INP_VEC_SIZE = ", INP_VEC_SIZE)
    
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
            batchFeats = utils.getFeatureVectorsFromDump(OFfeatures, keys, seqs, motion=True)
            #batchFeats = utils.getFeatureVectorsFromDump(HOGfeatures, keys, seqs, motion=False)
            #break

            # Training starts here
            inputs, target = utils.make_variables(batchFeats, labels, motion=True)
            #inputs, target = utils.make_variables(batchFeats, labels, motion=False)
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
    val_labs = [os.path.join(LABELS, f) for f in test_lab]
    val_sizes = [utils.getNFrames(os.path.join(DATASET, f+".avi")) for f in test_lst]
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
    OFValFeatures = utils.readAllOFfeatures(OFfeaturesPath, test_lst)
    #HOGValFeatures = utils.readAllHOGfeatures(HOGfeaturesPath, val_lst)   
    print("Predicting on the validation/test videos...")
    for i, (keys, seqs, labels) in enumerate(val_loader):
        
        # Testing on the sample
        #feats = getFeatureVectors(DATASET, keys, seqs)      # Parallelize this
        batchFeats = utils.getFeatureVectorsFromDump(OFValFeatures, keys, seqs, motion=True)
        #batchFeats = utils.getFeatureVectorsFromDump(HOGValFeatures, keys, seqs, motion=False)
        #break
        # Validation stage
        inputs, target = utils.make_variables(batchFeats, labels, motion=True)
        #inputs, target = utils.make_variables(batchFeats, labels, motion=False)
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

    # [4949, 4369, 4455, 4317, 4452]
    #predictions = [p.cpu() for p in predictions]  # convert to CPU tensor values
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
    filtered_shots = utils.filter_action_segments(localization_dict, epsilon=i)
    #i = 7  # optimum
    #filtered_shots = filter_non_action_segments(filtered_shots, epsilon=i)
    filt_shots_filename = "predicted_localizations_th0_5_filt"+str(i)+".json"
    with open(filt_shots_filename, 'w') as fp:
        json.dump(filtered_shots, fp)
    print("Prediction file written to disk !!")
    #####################################################################
