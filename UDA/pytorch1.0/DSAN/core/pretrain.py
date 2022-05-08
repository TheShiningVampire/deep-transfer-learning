"""Pre-train encoder and classifier for source dataset."""

import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm.auto import tqdm

import params
from utils import make_variable, save_model, EarlyStopping
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train_src(encoder, classifier, data_loader, data_loader_eval):
    """Train classifier for source domain."""
    ####################
    # 1. setup network #
    ####################
    # to track the average training loss per epoch as the model trains
    avg_train_losses = []
    # to track the average validation loss per epoch as the model trains
    avg_valid_losses = []

    valid_accuracies = []
    train_accuracies = []
    # initialize the early_stopping object
    patience = 7
    early_stopping2 = EarlyStopping(patience=patience, verbose=True)

    # set train state for Dropout and BN layers
    encoder = encoder.to(device)
    classifier = classifier.to(device)
    
    #encoder.train()
    #classifier.train()

    # setup criterion and optimizer
    optimizer = optim.Adam(
        list(encoder.parameters()) + list(classifier.parameters()),
        lr=0.0003,
        weight_decay=0.01)

    criterion = nn.CrossEntropyLoss()
    
    # lr scheduler with step decay
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.99)

    ####################
    # 2. train network #
    ####################

    for epoch in range(100):
        encoder.train()
        classifier.train()

        train_acc = float(0)
        # to track the training loss as the model trains
        train_losses = []
        # to track the validation loss as the model trains
        valid_losses = []

        for step, (images, labels) in tqdm(enumerate(data_loader)):
            # make images and labels variable
            images = make_variable(images)
            labels = make_variable(labels.squeeze_())

            # zero gradients for optimizer
            optimizer.zero_grad()
            
            with torch.set_grad_enabled(True):
                # compute loss for critic
                features = encoder(images)
                
                #print(features.shape)
                #features = features.to(device)
                features = features.reshape([len(labels), 1024])
                preds = classifier(features)
                loss = criterion(preds, labels)
                
                pred_clas = preds.data.max(1)[1]
                train_acc += pred_clas.eq(labels.data).cpu().sum()

                # optimize source classifier
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())
        
        train_acc /= float(len(data_loader.dataset))

        # eval model on test set
        valid_loss, valid_acc = eval_src(encoder, classifier, data_loader_eval)

        # lr scheduling 
        # scheduler.step()

        # checking early stopping
        #early_stopping2(valid_loss, classifier)

        # recording train/val accuracies
        train_accuracies.append(train_acc)
        valid_accuracies.append(valid_acc)

        # recording train/val losses
        train_loss = np.average(train_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)
        
        # print step info
        print("Epoch [{}] Step [{}]: train_loss={}, train_Acc={:2%}, val_loss={}, val_Acc={:2%}"
                .format(epoch+1,
                        step+1,
                        train_loss,
                        train_acc,
                        valid_loss,
                        valid_acc))


        
        if (early_stopping2.early_stop):
            save_model(encoder, "KimiaNet-ADDA-source-encoder-ES-5-{}.pt".format(epoch + 1))
            save_model(classifier, "KimiaNet-ADDA-source-classifier-ES-5-{}.pt".format(epoch + 1))
            print("Early stopping")
            break
        

        # save model parameters after every 50 epochs
        elif ((epoch + 1) % 10 == 0):
            save_model(encoder, "KimiaNet-ADDA-source-encoder-5-{}.pt".format(epoch + 1))
            save_model(
                classifier, "KimiaNet-ADDA-source-classifier-5-{}.pt".format(epoch + 1))
    

    # # save final model
    save_model(encoder, "KimiaNet-ADDA-source-encode-5-final.pt")
    save_model(classifier, "KimiaNet-ADDA-source-classifier-5-final.pt")

    return encoder, classifier, avg_train_losses, avg_valid_losses, train_accuracies, valid_accuracies


def eval_src(encoder, classifier, data_loader):
    """Evaluate classifier for source domain."""
    # set eval state for Dropout and BN layers
    encoder = encoder.to(device)
    classifier = classifier.to(device) 

    encoder.eval()
    classifier.eval()

    # init loss and accuracy
    valid_losses = []
    acc = float(0)

    # set loss function
    criterion = nn.CrossEntropyLoss()

    # evaluate network
    for (images, labels) in tqdm(data_loader):
        images = make_variable(images, volatile=True)
        labels = make_variable(labels)
        feature = encoder(images)
        feature = feature.reshape([len(labels), 1024])
        #feature = feature.to(device)

        preds = classifier(feature)
        loss = criterion(preds, labels)
        valid_losses.append(loss.item())

        pred_cls = preds.data.max(1)[1]
        acc += pred_cls.eq(labels.data).cpu().sum()
        # record validation loss

    valid_loss = np.average(valid_losses)
    acc /= float(len(data_loader.dataset))

    print("val_loss = {}, val_Accuracy = {:2%}".format(valid_loss, acc))
    return valid_loss, acc
