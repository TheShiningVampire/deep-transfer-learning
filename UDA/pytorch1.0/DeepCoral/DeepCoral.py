from __future__ import print_function
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import os
import math
import data_loader
import ResNet as models
from torch.utils import model_zoo
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Training settings
batch_size = 16
iteration=1000
lr = [0.001, 0.01]
momentum = 0.9
no_cuda =False
seed = 8
log_interval = 1
l2_decay = 5e-4
root_path = "./dataset/Bach_Breakhis/"
src_name = "Breakhis"
tgt_name = "Bach"

cuda = not no_cuda and torch.cuda.is_available()

torch.manual_seed(seed)
if cuda:
    torch.cuda.manual_seed(seed)

kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}

src_loader = data_loader.load_training(root_path, src_name, batch_size, kwargs)
tgt_train_loader = data_loader.load_training(root_path, tgt_name, batch_size, kwargs)
tgt_test_loader = data_loader.load_testing(root_path, tgt_name, batch_size, kwargs)

src_dataset_len = len(src_loader.dataset)
tgt_dataset_len = len(tgt_test_loader.dataset)
src_loader_len = len(src_loader)
tgt_loader_len = len(tgt_train_loader)


def train(model):
    src_iter = iter(src_loader)
    tgt_iter = iter(tgt_train_loader)
    correct=0

    optimizer = torch.optim.SGD([
        {'params': model.sharedNet.parameters()},
        {'params': model.cls_fc.parameters(), 'lr': lr[1]},
        ], lr=lr[0], momentum=momentum, weight_decay=l2_decay)

    for i in range(1, iteration+1):
        model.train()
        for index, param_group in enumerate(optimizer.param_groups):
            param_group['lr'] = lr[index] / math.pow((1 + 10 * (i - 1) / iteration), 0.75)

        try:
            src_data, src_label = src_iter.next()
        except Exception as err:
            src_iter=iter(src_loader)
            src_data, src_label = src_iter.next()
            
        try:
            tgt_data, _ = tgt_iter.next()
        except Exception as err:
            tgt_iter=iter(tgt_train_loader)
            tgt_data, _ = tgt_iter.next()
            
        if cuda:
            src_data, src_label = src_data.cuda(), src_label.cuda()
            tgt_data = tgt_data.cuda()

        optimizer.zero_grad()
        src_pred, coral_loss = model(src_data, tgt_data)
        cls_loss = F.nll_loss(F.log_softmax(src_pred, dim=1), src_label)
        lambd = 2 / (1 + math.exp(-10 * (i) / iteration)) - 1
        loss = cls_loss + lambd * coral_loss
        loss.backward()
        optimizer.step()
        if i % log_interval == 0:
            print('Train iter: {} [({:.0f}%)]\ttotal_Loss: {:.6f}\tcls_Loss: {:.6f}\tcoral_Loss: {:.6f}'.format(
                i, 100. * i / iteration, loss.item(), cls_loss.item(), coral_loss.item()))

        if i%(log_interval*20)==0:
            t_correct = test(model)
            if t_correct > correct:
                correct = t_correct
            print('src: {} to tgt: {} max correct: {} max accuracy{: .2f}%\n'.format(
              src_name, tgt_name, correct, 100. * correct / tgt_dataset_len ))

def test(model):
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for tgt_test_data, tgt_test_label in tgt_test_loader:
            if cuda:
                tgt_test_data, tgt_test_label = tgt_test_data.cuda(), tgt_test_label.cuda()
            tgt_test_data, tgt_test_label = Variable(tgt_test_data), Variable(tgt_test_label)
            tgt_pred, coral_loss = model(tgt_test_data, tgt_test_data)
            test_loss += F.nll_loss(F.log_softmax(tgt_pred, dim = 1), tgt_test_label, reduction='sum').item() # sum up batch loss
            pred = tgt_pred.data.max(1)[1] # get the index of the max log-probability
            correct += pred.eq(tgt_test_label.data.view_as(pred)).cpu().sum()

    test_loss /= tgt_dataset_len
    print('\n{} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        tgt_name, test_loss, correct, tgt_dataset_len,
        100. * correct / tgt_dataset_len))
    return correct


if __name__ == '__main__':
    model = models.DeepCoral(num_classes=2)
    print(model)
    if cuda:
        model.cuda()
    train(model)
