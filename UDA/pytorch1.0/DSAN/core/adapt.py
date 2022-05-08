"""Adversarial adaptation to train target encoder."""

import os
import numpy as np
import torch
import torch.optim as optim
from torch import nn
from tqdm.auto import tqdm

import params
from utils import make_variable
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def eval_src(encoder, classifier, data_loader):
        """Evaluate classifier for source domain."""
        # set eval state for Dropout and BN layers
        
        #encoder = encoder.to(device)
        #classifier = classifier.to(device)

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
            preds = classifier(feature)
            loss = criterion(preds, labels)
            valid_losses.append(loss.item())

            pred_cls = preds.data.max(1)[1]
            acc += pred_cls.eq(labels.data).cpu().sum()
        
        # record validation loss
        valid_loss = np.average(valid_losses)
        acc /= float(len(data_loader.dataset))

        print("val_loss = {}, val_Accuracy = {:2%}".format(loss, acc))
        return valid_loss, acc



def train_tgt(src_encoder, tgt_encoder, critic,
              src_data_loader, tgt_data_loader, src_classifier, tgt_data_loader_eval):
    """Train encoder for target domain."""
    ####################
    # 1. setup network #
    ####################
    tgt_encoder = tgt_encoder.to(device)
    src_encoder = src_encoder.to(device)
    critic = critic.to(device)
    src_classifier = src_classifier.to(device)

    # set train state for Dropout and BN layers
    tgt_encoder.train()
    critic.train()

    # setup criterion and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer_tgt = optim.Adam(tgt_encoder.parameters(),
                               lr=3e-4,
                               weight_decay=0.01)
    
    optimizer_critic = optim.Adam(critic.parameters(),
                                  lr=3e-4,
                                  weight_decay=0.01)

    len_data_loader = min(len(src_data_loader), len(tgt_data_loader))

    ####################
    # 2. train network #
    ####################
    d_losses = []
    g_losses = []
    d_acc_avg = []
    max_acc = 0

    for epoch in range(1000):
        # zip source and target data pair
        data_zip = enumerate(zip(src_data_loader, tgt_data_loader))
        d_loss = []
        g_loss = []
        d_acc = []
        for step, ((images_src, _), (images_tgt, _)) in tqdm(data_zip):
            ###########################
            # 2.1 train discriminator #
            ###########################

            # make images variable
            images_src = make_variable(images_src)
            images_tgt = make_variable(images_tgt)

            # training discriminator
            with torch.set_grad_enabled(True):
                # zero gradients for optimizer

                optimizer_critic.zero_grad()
                
                # extract and concat features
                feat_src = src_encoder(images_src)
                feat_tgt = tgt_encoder(images_tgt)
            
                feat_src = feat_src.reshape([len(images_src), 1024])
                feat_tgt = feat_tgt.reshape([len(images_tgt), 1024])
            
                feat_concat = torch.cat((feat_src, feat_tgt), 0)

                # predict on discriminator
                pred_concat = critic(feat_concat.detach())

                # prepare real and fake label
                # src label: 1, tgt label: 0
                label_src = make_variable(torch.ones(feat_src.size(0)).long())
                label_tgt = make_variable(torch.zeros(feat_tgt.size(0)).long())
                label_concat = torch.cat((label_src, label_tgt), 0)

                # compute loss for critic
                loss_critic = criterion(pred_concat, label_concat)
                loss_critic.backward()

                # optimize critic
                optimizer_critic.step()

                pred_cls = torch.squeeze(pred_concat.max(1)[1])
                acc = (pred_cls == label_concat).float().mean()
            
                d_acc.append(acc.item())
                d_loss.append(loss_critic.item())

            
                ############################
                # 2.2 train target encoder #
                ############################
           
                # train generator
                g_l = float(0)
                for i in range(10):
                    # zero gradients for optimizer
                    optimizer_critic.zero_grad()
                    optimizer_tgt.zero_grad()

                    # extract and target features
                    feat_tgt = tgt_encoder(images_tgt)
                    feat_tgt = feat_tgt.reshape([len(images_tgt), 1024])

                    # predict on discriminator
                    pred_tgt = critic(feat_tgt)

                    # prepare fake labels
                    label_tgt = make_variable(torch.ones(feat_tgt.size(0)).long())
 
                    # compute loss for target encoder
                    loss_tgt = criterion(pred_tgt, label_tgt)
                    loss_tgt.backward()

                    # optimize target encoder
                    optimizer_tgt.step()
                    g_l += loss_tgt.item()

                g_loss.append(g_l/10.0)


        """
        # print step update for discriminator
        if(epoch%5==0):
            d_loss_1 = np.average(d_loss)
            d_acc_1 = np.average(d_acc)
            d_losses.append(d_loss_1)
            d_acc_avg.append(d_acc_1)

            print("=== Discriminator updated ===")
            print("Epoch [{}/{}], d_loss={}, Dis_Acc={:2%}"
                                          .format(epoch + 1,
                                                  500,
                                                  d_loss_1,
                                                  d_acc_1))
        
        # print step update for generator
        if(True):
            g_loss_1 = np.average(g_loss)
            g_losses.append(g_loss_1)

            print("=== Generator updated ===")
            print("Epoch [{}/{}], g_loss={}"
                                         .format(epoch + 1,
                                                 500,
                                                 g_loss_1))
        
            
        """ 

        d_loss_1 = np.average(d_loss)
        g_loss_1 = np.average(g_loss)
        d_acc_1 = np.average(d_acc)

        d_losses.append(d_loss_1)
        g_losses.append(g_loss_1)
        d_acc_avg.append(d_acc_1)
        

        #######################
        # 2.3 print step info #
        #######################

        print("Epoch [{}/{}], d_loss={}, g_loss={}, Dis_Acc={:2%}"
                      .format(epoch + 1,
                              1000,
                              d_loss_1,
                              g_loss_1,
                              d_acc_1))

        print(" ")
        print(" ")


        if(True):
            print("==== tgt evaluation ====")
            _, curr_acc = eval_src(tgt_encoder, src_classifier, tgt_data_loader_eval)
            print(" ")
            print(" ")
        

            #############################
            # 2.4 save model parameters #
            #############################
            if (curr_acc == max(curr_acc, max_acc)):
                max_acc = curr_acc
                torch.save(critic.state_dict(), os.path.join(
                    params.model_root,
                    "KimiaNet-ADDA-critic-14-{}-{}.pt".format(epoch + 1, curr_acc)))
                torch.save(tgt_encoder.state_dict(), os.path.join(
                    params.model_root,
                    "KimiaNet-ADDA-target-encoder-14-{}-{}.pt".format(epoch + 1, curr_acc)))

    torch.save(critic.state_dict(), os.path.join(
        params.model_root,
        "KimiaNet-ADDA-critic-14-final.pt"))
    torch.save(tgt_encoder.state_dict(), os.path.join(
        params.model_root,
        "KimiaNet-ADDA-target-encoder-14-final.pt"))
    
    return tgt_encoder, d_losses, g_losses, d_acc_avg
    #return tgt_encoder, 10, g_losses, 10

