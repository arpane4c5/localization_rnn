#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 8 01:53:27 2018

@author: Arpan

@Description: Finetune a pretrained C3D model model in PyTorch. 
Use the highlight videos dataset and re-trained the FC7 layer.
"""

import torch
import numpy as np
import os
import torch.nn as nn
import model_c3d_finetune as c3d
from torch.utils.data import DataLoader
#import matplotlib.pyplot as plt
import pickle
import time
import utils

from Video_Dataset import VideoDataset
from model_gru import RNNClassifier
from torch.autograd import Variable
from glob import glob
import copy


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
BATCH_SIZE = 32
N_EPOCHS = 2
INP_VEC_SIZE = None
SEQ_SIZE = 16   # has to >=16 (ie. the number of frames used for c3d input)
threshold = 0.5
seq_threshold = 0.5


#def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
#    since = time.time()
#
#    best_model_wts = copy.deepcopy(model.state_dict())
#    best_acc = 0.0
#
#    for epoch in range(num_epochs):
#        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
#        print('-' * 10)
#
#        # Each epoch has a training and validation phase
#        for phase in ['train', 'val']:
#            if phase == 'train':
#                scheduler.step()
#                model.train()  # Set model to training mode
#            else:
#                model.eval()   # Set model to evaluate mode
#
#            running_loss = 0.0
#            running_corrects = 0
#
#            # Iterate over data.
#            for inputs, labels in dataloaders[phase]:
#                inputs = inputs.to("cpu")
#                labels = labels.to("cpu")
#
#                # zero the parameter gradients
#                optimizer.zero_grad()
#
#                # forward
#                # track history if only in train
#                with torch.set_grad_enabled(phase == 'train'):
#                    outputs = model(inputs)
#                    _, preds = torch.max(outputs, 1)
#                    loss = criterion(outputs, labels)
#
#                    # backward + optimize only if in training phase
#                    if phase == 'train':
#                        loss.backward()
#                        optimizer.step()
#
#                # statistics
#                running_loss += loss.item() * inputs.size(0)
#                running_corrects += torch.sum(preds == labels.data)
#
#            epoch_loss = running_loss / dataset_sizes[phase]
#            epoch_acc = running_corrects.double() / dataset_sizes[phase]
#
#            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
#                phase, epoch_loss, epoch_acc))
#
#            # deep copy the model
#            if phase == 'val' and epoch_acc > best_acc:
#                best_acc = epoch_acc
#                best_model_wts = copy.deepcopy(model.state_dict())
#
#        print()
#
#    time_elapsed = time.time() - since
#    print('Training complete in {:.0f}m {:.0f}s'.format(
#        time_elapsed // 60, time_elapsed % 60))
#    print('Best val Acc: {:4f}'.format(best_acc))
#
#    # load best model weights
#    model.load_state_dict(best_model_wts)
#    return model

def getBatchFrames(features, videoFiles, sequences):
    """Select only the batch features from the dictionary of features (corresponding
    to the given sequences) and return them as a list of lists. 
    OFfeatures: a dictionary of features {vidname: numpy matrix, ...}
    videoFiles: the list of filenames for a batch
    sequences: the start and end frame numbers in the batch videos to be sampled.
    SeqSize should be >= 2 for atleast one vector in sequence.
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
        
        # get depth x 112 x 112 x 3 sized input cubiod
        vid_feat_seq = vidFeats[start_frame:(end_frame+1), :, :]
        
        # transpose to Ch x depth x H x W
        #vid_feat_seq = vid_feat_seq.transpose(3, 0, 1, 2)
        #vid_feat_seq = np.squeeze(vid_feat_seq, axis = 1)
        batch_feats.append(vid_feat_seq)
        
    return np.array(batch_feats)


# Inputs: feats: list of lists
def make_variables(feats, labels):
    # Create the input tensors and target label tensors
    # transpose to batch x ch x depth x H x W
    feats = feats.transpose(0, 4, 1, 2, 3)    
    feats = torch.from_numpy(np.float32(feats))
    
    #feats = torch.Tensor(np.array(feats).astype(np.float32))
    feats[feats==float("-Inf")] = 0
    feats[feats==float("Inf")] = 0
    # Form the target labels 
    target = []
    # Append the sequence of labels of len (seq_size-1) to the target list for OF.
    # Iterate over the batch labels, for each extract seq_size labels and extend 
    # in the target list
    seq_size = len(labels)
    
    for i in range(labels[0].size(0)):
        lbls = [y[i] for y in labels]      # get labels of frames (size seq_size)
        if sum(lbls)>=8:
            target.extend(1)
        else:
            target.extend(0)

    # Form a wrap into a tensor variable as B X S X I
    # target is a vector of batchsize
    return utils.create_variable(feats), utils.create_variable(torch.Tensor(target))

if __name__=='__main__':

#    #model_parameters = filter(lambda p: p.requires_grad, model.parameters())
#    #params = sum([np.prod(p.size()) for p in model_parameters])
#    # or call count_paramters(model)  
#    print "#Parameters : {} ".format(count_parameters(model))
#
#    ###########################################################################
#    # read labels
#    labels = utils.read_labels_from_file('labels.txt')
#
#    # load a clip to be predicted
#    clip_name = "ICC WT20 Australia vs Bangladesh - Match Highlights"
#    N = 16
#    #X = utils.get_sport_clip('TaiChi/v_TaiChi_g18_c01', verbose=False)
#    
#    frames_list = sorted(glob(os.path.join('data', clip_name, '*.jpg')))
#    totalFrames = len(frames_list)
#    # get network pretrained model
#    model.load_state_dict(torch.load('c3d.pickle'))
#    #model.cuda()
#    model.eval()
#
#    # call for each 
#    for i in range(len(frames_list)-N+1):
#        X = utils.get_sport_clip(frames_list[i:i+N], verbose = False)
#        X = Variable(X)
#        #X = X.cuda()
#
#        # perform prediction
#        prediction = model(X)
#        prediction = prediction.data.cpu().numpy()
#
#        # print top predictions
#        top_inds = prediction[0].argsort()[::-1][:5]  # reverse sort and take five largest items
#        print('\nTop 5: {} / {}'.format(i+1, totalFrames))
#        for i in top_inds:
#            print('{:.5f} {}'.format(prediction[0][i], labels[i]))
            
#    #####################################################################
#    #####################################################################

    # Divide the samples files into training set, validation and test sets
    train_lst, val_lst, test_lst = utils.split_dataset_files(DATASET)
    print(train_lst, len(train_lst))
    print 60*"-"
    
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
    
    framesPath = os.path.join(os.getcwd(),"numpy_vids_112x112")
    
    #####################################################################
    
    # Create a DataLoader object and sample batches of examples. 
    # These batch samples are used to extract the features from videos parallely
    train_loader = DataLoader(dataset=hlDataset, batch_size=BATCH_SIZE, shuffle=True)

    train_losses = []
    # read into dictionary {vidname: np array, ...}
    print("Loading features from disk...")
    # load matrices of size N x H x W (N is #frames H W are cropped height and width)
    frames = utils.readAllNumpyFrames(framesPath, train_lst)
    #HOGfeatures = utils.readAllHOGfeatures(HOGfeaturesPath, train_lst)
    #features = utils.readAllPartitionFeatures(featuresPath, train_lst)
    print(len(train_loader.dataset))
    
    #####################################################################
    
    model = c3d.C3D()
    # get the network pretrained weights into the model
    model.load_state_dict(torch.load('c3d.pickle'))
#    # need to set requires_grad = False for all the layers
    for param in model.parameters():
        param.requires_grad = False
    # reset the last layer
    model.fc8 = nn.Linear(4096, 2)
    # Load on the GPU, if available
#    if torch.cuda.device_count() > 1:
#        print("Let's use", torch.cuda.device_count(), "GPUs!")
#        # Parallely run on multiple GPUs using DataParallel
#        model = nn.DataParallel(model)
#
#    if torch.cuda.is_available():
#        model.cuda()    
    
    #####################################################################
    # set the scheduler, optimizer and retrain
        
    #####################################################################
    
    #fc7 layer output size
#    INP_VEC_SIZE = features[features.keys()[0]].shape[-1] 
#    print("INP_VEC_SIZE = ", INP_VEC_SIZE)
    
    # Creating the RNN and training
#    classifier = RNNClassifier(INP_VEC_SIZE, HIDDEN_SIZE, 1, N_LAYERS)


    #optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    optimizer = torch.optim.SGD(model.fc8.parameters(), lr=0.001, momentum=0.9)
    
    sigm = nn.Sigmoid()
    #criterion = nn.BCELoss()
    # Decay LR by a factor of 0.1 every 7 epochs
    #exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    
    start = time.time()
#    model_ft = train_model(model, criterion, optimizer, exp_lr_scheduler,
#                       num_epochs=25)
    
    print("Training for %d epochs..." % N_EPOCHS)
    for epoch in range(N_EPOCHS):
        total_loss = 0
        for i, (keys, seqs, labels) in enumerate(train_loader):
            # Run your training process
            #print(epoch, i) #, "keys", keys, "Sequences", seqs, "Labels", labels)
            #feats = getFeatureVectors(DATASET, keys, seqs)   # Takes time. Do not use
            
            # return a 256 x ch x depth x H x W
            batchFeats = getBatchFrames(frames, keys, seqs)
            
            inputs, target = make_variables(batchFeats, labels)
            
            output = model(inputs)

            #loss = criterion(sigm(output.view(output.size(0))), target)
            loss = criterion(output.view(output.size(1)), target)
            total_loss += loss.data[0]

            model.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 2 == 0:
                print('Train Epoch: {} :: Loss: {:.2f}'.format(epoch, total_loss))
            #if (i+1) % 10 == 0:
            #    break
        train_losses.append(total_loss)
        print('Train Epoch: {} :: Loss: {:.2f}'.format(epoch+1, total_loss))
    
    
#    # Save only the model params
#    torch.save(classifier.state_dict(), "gru_100_epoch10_BCE.pt")
#    print "Model saved to disk..."
#    
#    # Save losses to a txt file
#    with open("losses.pkl", "w") as fp:
#        pickle.dump(train_losses, fp)
    
    # To load the params into model
    ##the_model = RNNClassifier(INP_VEC_SIZE, HIDDEN_SIZE, 1, N_LAYERS)
    ##the_model.load_state_dict(torch.load("gru_100_epoch10_BCE.pt"))    
    #classifier.load_state_dict(torch.load("gru_100_epoch10_BCE.pt"))
    
    #####################################################################
#    
#    # Test a video or calculate the accuracy using the learned model
#    print "Prediction video meta info."
#    val_labs = [os.path.join(LABELS, f) for f in val_lab]
#    val_sizes = [utils.getNFrames(os.path.join(DATASET, f+".avi")) for f in val_lst]
#    print "Size : {}".format(val_sizes)
#    hlvalDataset = VideoDataset(val_labs, val_sizes, seq_size=SEQ_SIZE, is_train_set = False)
#    print hlvalDataset.__len__()
#    
#    # Create a DataLoader object and sample batches of examples. 
#    # These batch samples are used to extract the features from videos parallely
#    val_loader = DataLoader(dataset=hlvalDataset, batch_size=BATCH_SIZE, shuffle=False)
#    print(len(val_loader.dataset))
#    correct = 0
#    val_keys = []
#    predictions = []
#    print "Loading validation/test features from disk..."
#    #OFValFeatures = utils.readAllOFfeatures(OFfeaturesPath, test_lst)
#    #HOGValFeatures = utils.readAllHOGfeatures(HOGfeaturesPath, val_lst)   
#    valFeatures = utils.readAllPartitionFeatures(featuresPath, val_lst)
#    print("Predicting on the validation/test videos...")
#    for i, (keys, seqs, labels) in enumerate(val_loader):
#        
#        # Testing on the sample
#        #feats = getFeatureVectors(DATASET, keys, seqs)      # Parallelize this
#        #batchFeats = utils.getFeatureVectorsFromDump(OFValFeatures, keys, seqs, motion=True)
#        #batchFeats = utils.getFeatureVectorsFromDump(HOGValFeatures, keys, seqs, motion=False)
#        batchFeats = utils.getC3DFeatures(valFeatures, keys, seqs)
#        #break
#        # Validation stage
#        inputs, target = utils.make_c3d_variables(batchFeats, labels)
#        #inputs, target = utils.make_variables(batchFeats, labels, motion=False)
#        output = classifier(inputs) # of size (BATCHESxSeqLen) X 1
#
#        #pred = output.data.max(1, keepdim=True)[1]  # get max value in each row
#        pred_probs = sigm(output.view(output.size(0))).data  # get the normalized values (0-1)
#        #preds = pred_probs > THRESHOLD  # ByteTensor
#        #correct += pred.eq(target.data.view_as(pred)).cpu().sum()
#        val_keys.append(keys)
#        predictions.append(pred_probs)  # append the 
#        
#        #loss = criterion(m(output.view(output.size(0))), target)
#        #total_loss += loss.data[0]
#
#        #if i % 2 == 0:
#        #    print('i: {} :: Val keys: {} : seqs : {}'.format(i, keys, seqs)) #keys, pred_probs))
#        #if (i+1) % 10 == 0:
#        #    break
#    print "Predictions done on validation/test set..."
#    #####################################################################
#    
#    with open("predictions.pkl", "wb") as fp:
#        pickle.dump(predictions, fp)
#    
#    with open("val_keys.pkl", "wb") as fp:
#        pickle.dump(val_keys, fp)
#    
##    with open("predictions.pkl", "rb") as fp:
##        predictions = pickle.load(fp)
##    
##    with open("val_keys.pkl", "rb") as fp:
##        val_keys = pickle.load(fp)
#    
#    from get_localizations import getLocalizations
#    from get_localizations import getVidLocalizations
#
#    # [4949, 4369, 4455, 4317, 4452]
#    #predictions = [p.cpu() for p in predictions]  # convert to CPU tensor values
#    localization_dict = getLocalizations(val_keys, predictions, BATCH_SIZE, \
#                                         threshold, seq_threshold)
#
#    print localization_dict
#    
#    import json        
##    for i in range(0,101,10):
##        filtered_shots = filter_action_segments(localization_dict, epsilon=i)
##        filt_shots_filename = "predicted_localizations_th0_5_filt"+str(i)+".json"
##        with open(filt_shots_filename, 'w') as fp:
##            json.dump(filtered_shots, fp)
#
#    # Apply filtering    
#    i = 60  # optimum
#    filtered_shots = utils.filter_action_segments(localization_dict, epsilon=i)
#    #i = 7  # optimum
#    #filtered_shots = filter_non_action_segments(filtered_shots, epsilon=i)
#    filt_shots_filename = "predicted_localizations_th0_5_filt"+str(i)+".json"
#    with open(filt_shots_filename, 'w') as fp:
#        json.dump(filtered_shots, fp)
#    print("Prediction file written to disk !!")
#    #####################################################################
#    # count no. of parameters in the model
#    #model_parameters = filter(lambda p: p.requires_grad, model.parameters())
#    #params = sum([np.prod(p.size()) for p in model_parameters])
#    # or call count_paramters(model)  
#    print "#Parameters : {} ".format(utils.count_parameters(classifier))
