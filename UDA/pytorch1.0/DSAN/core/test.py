"""Test script to classify target data."""

import torch
import torch.nn as nn

from utils import make_variable
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def eval_tgt(encoder, classifier, data_loader):
    """Evaluation for target encoder by source classifier on target dataset."""
    # set eval state for Dropout and BN layers
    encoder.eval()
    classifier.eval()
    
    encoder = encoder.to(device)
    classifier = classifier.to(device)

    # init loss and accuracy
    loss = 0
    acc = 0

    # set loss function
    criterion = nn.CrossEntropyLoss()

    # evaluate network
    for (images, labels) in data_loader:
        images = make_variable(images, volatile=True)
        labels = make_variable(labels).squeeze_()
        feat = encoder(images)
        feat = feat.reshape([1, 1024])
        preds = classifier(feat)
        loss += criterion(preds, labels).item()

        pred_cls = preds.data.max(1)[1]
        acc += pred_cls.eq(labels.data).cpu().sum()

    loss /= len(data_loader)
    acc /= len(data_loader.dataset)

    print("Avg Loss = {}, Avg Accuracy = {:2%}".format(loss, acc))
