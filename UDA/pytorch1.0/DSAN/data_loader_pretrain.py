from torchvision import datasets, transforms
import torch
import os

def load_pretrain_data(root_path, dir, batch_size, train = True):

    if train:
        transform = transforms.Compose([
        # transforms.Resize([256, 256]),
        #  transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    else:
        transform = transforms.Compose([
        # transforms.Resize([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


    data = datasets.ImageFolder(root=os.path.join(root_path, dir), transform=transform)
    data_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4)
    return data_loader
