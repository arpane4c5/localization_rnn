#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 21 13:53:27 2018

@author: Arpan

@Description: C3D model implementation in PyTorch. Learn cricket strokes.
"""

import torch
import numpy as np
import cv2
import os
import torch.nn as nn
from torch.utils.data import DataLoader
#import matplotlib.pyplot as plt
import pickle
import time
import utils

from Video_Dataset import VideoDataset
from model_gru import RNNClassifier
from torch.autograd import Variable
from glob import glob
import skimage.io as io
from skimage.transform import resize


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
SEQ_SIZE = 17   # has to >=16 (ie. the number of frames used for c3d input)
threshold = 0.5
seq_threshold = 0.5


def train_model(model, destpath):
    train_files = sorted(os.listdir(DATASET))
    labels = sorted(os.listdir(LABELS))
    
    print train_files
    print labels

# function to count the number of parameters in the model
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



def get_sport_clip(frames_list, verbose=True):
    """
    Loads a clip to be fed to C3D for classification.
    TODO: should I remove mean here?
    
    Parameters
    ----------
    clip_name: str      OR frames_list : list of consecutive N framePaths
        the name of the clip (subfolder in 'data').
    verbose: bool
        if True, shows the unrolled clip (default is True).

    Returns
    -------
    Tensor
        a pytorch batch (n, ch, fr, h, w).
    """

    #clip = sorted(glob(os.path.join('data', clip_name, '*.jpg')))
    clip = np.array([resize(io.imread(frame), output_shape=(112, 200), preserve_range=True) for frame in frames_list])
    clip = clip[:, :, 44:44+112, :]  # crop centrally

    if verbose:
        clip_img = np.reshape(clip.transpose(1, 0, 2, 3), (112, 16 * 112, 3))
        io.imshow(clip_img.astype(np.uint8))
        io.show()

    clip = clip.transpose(3, 0, 1, 2)  # ch, fr, h, w
    clip = np.expand_dims(clip, axis=0)  # batch axis
    clip = np.float32(clip)

    return torch.from_numpy(clip)


def read_labels_from_file(filepath):
    """
    Reads Sport1M labels from file
    
    Parameters
    ----------
    filepath: str
        the file.
        
    Returns
    -------
    list
        list of sport names.
    """
    with open(filepath, 'r') as f:
        labels = [line.strip() for line in f.readlines()]
    return labels



if __name__=='__main__':
    
#    # create a model 
#    import model_c3d as c3d
#    
#    model = c3d.C3D()
#    
#    ###########################################################################
#    #print model
#    train_model(model, "data")
#    
#    # count no. of parameters in the model
#    #model_parameters = filter(lambda p: p.requires_grad, model.parameters())
#    #params = sum([np.prod(p.size()) for p in model_parameters])
#    # or call count_paramters(model)  
#    print "#Parameters : {} ".format(count_parameters(model))
#    
#    # Creation of a training set, validation set, test set meta info file.
#    
#    # Train the model
#    
#    # 
#
#    ###########################################################################
#    # read labels
#    labels = read_labels_from_file('labels.txt')
#
#    # load a clip to be predicted
#    clip_name = "ICC WT20 Australia vs Bangladesh - Match Highlights"
#    N = 16
#    #X = get_sport_clip('TaiChi/v_TaiChi_g18_c01', verbose=False)
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
#        X = get_sport_clip(frames_list[i:i+N], verbose = False)
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
    # Extract the frames as per the sequences, bunch them and save them to disk
    
    # Run extract_denseOF_par.py before executing this file, using same grid_size
    # Features already extracted and dumped to disk 
    # Read those features using the given path and grid size
    #framesPath = os.path.join(os.getcwd(),"numpy_frames_cropped")
    #HOGfeaturesPath = os.path.join(os.getcwd(),"hog_feats_new")
    c3dFeaturesPath = os.path.join(os.getcwd(),"c3d_feats_16")
    
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
    #frames = utils.readAllNumpyFrames(framesPath, train_lst)
    #HOGfeatures = utils.readAllHOGfeatures(HOGfeaturesPath, train_lst)
    c3dfeats = utils.readAllC3DFeatures(c3dFeaturesPath, train_lst)
    print(len(train_loader.dataset))
    
    ########
    
    INP_VEC_SIZE = 4096 # change to the fc7 layer output size
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
            # get a list of seq_size x 4096 matrices, for this batch. 
            # no. of matrices in list is the size of the batch
            batchFeats = utils.getC3DFeatures(c3dfeats, keys, seqs)
            #batchFeats = utils.getFeatureVectorsFromDump(HOGfeatures, keys, seqs, motion=False)
            #break

            # Training starts here
            inputs, target = utils.make_c3d_variables(batchFeats, labels)
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
    val_labs = [os.path.join(LABELS, f) for f in val_lab]
    val_sizes = [utils.getNFrames(os.path.join(DATASET, f+".avi")) for f in val_lst]
    print "Size : {}".format(val_sizes)
    hlvalDataset = VideoDataset(val_labs, val_sizes, seq_size=SEQ_SIZE, is_train_set = False)
    print hlvalDataset.__len__()
    
    # Create a DataLoader object and sample batches of examples. 
    # These batch samples are used to extract the features from videos parallely
    val_loader = DataLoader(dataset=hlvalDataset, batch_size=BATCH_SIZE, shuffle=False)
    print(len(val_loader.dataset))
    correct = 0
    val_keys = []
    predictions = []
    print "Loading validation/test features from disk..."
    #OFValFeatures = utils.readAllOFfeatures(OFfeaturesPath, test_lst)
    #HOGValFeatures = utils.readAllHOGfeatures(HOGfeaturesPath, val_lst)   
    c3dValFeatures = utils.readAllC3DFeatures(c3dFeaturesPath, val_lst)
    print("Predicting on the validation/test videos...")
    for i, (keys, seqs, labels) in enumerate(val_loader):
        
        # Testing on the sample
        #feats = getFeatureVectors(DATASET, keys, seqs)      # Parallelize this
        #batchFeats = utils.getFeatureVectorsFromDump(OFValFeatures, keys, seqs, motion=True)
        #batchFeats = utils.getFeatureVectorsFromDump(HOGValFeatures, keys, seqs, motion=False)
        batchFeats = utils.getC3DFeatures(c3dValFeatures, keys, seqs)
        #break
        # Validation stage
        inputs, target = utils.make_c3d_variables(batchFeats, labels)
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
