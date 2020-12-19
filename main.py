import os
import torch
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import time
import copy

from torch.optim import lr_scheduler
from torchvision import datasets
from efficientnet_pytorch import EfficientNet

# from torchsummary import summary
from efficientnet_pytorch import EfficientNet

model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=17)

# summary(model, input_size=(3, 331, 331), device='cpu')

batch_size = 16
epochs = 30
data_dir = '../DL_Final/barkSNU/'
# writer = SummaryWriter('./runs/experiment1/')
test_split = 0.25

data_transforms = {'train': transforms.Compose([
    transforms.Resize(200, 200),
    transforms.RandomRotation(30),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]),
    'val': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.Resize(200, 200)
    ])}

from torch.utils.data import Dataset, DataLoader

import natsort
from torch.utils.data import random_split
from PIL import Image

from torch.utils.data import Dataset


class ApplyTransform(Dataset):
    """
    Apply transformations to a Dataset

    Arguments:
        dataset (Dataset): A Dataset that returns (sample, target)
        transform (callable, optional): A function/transform to be applied on the sample
        target_transform (callable, optional): A function/transform to be applied on the target

    """

    def __init__(self, dataset, transform=None, target_transform=None):
        self.dataset = dataset
        self.transform = transform
        self.target_transform = target_transform
        # yes, you don't need these 2 lines below :(
        if transform is None and target_transform is None:
            print("Am I a joke to you? :)")

    def __getitem__(self, idx):
        sample, target = self.dataset[idx]
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target

    def __len__(self):
        return len(self.dataset)


import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
from torchvision.transforms import Compose, ToTensor, Resize
from torch.utils.data import DataLoader

import numpy

def train_val_dataset(dataset, val_split=0.25):
    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=val_split)
    datasets = {}
    datasets['train'] = Subset(dataset, train_idx)
    datasets['val'] = Subset(dataset, val_idx)
    return datasets


dataset = ImageFolder(data_dir, transform=Compose(
    [Resize((200, 200)), ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]))

dataset_subset = torch.utils.data.Subset(dataset, numpy.random.choice(len(dataset), 5000, replace=False))

print(len(dataset_subset))
datasets = train_val_dataset(dataset_subset)
# The original dataset is available in the Subset class
print(datasets['train'].dataset)
# datasets['train'] = ApplyTransform(datasets['train'], transforms.RandomRotation(30))
# datasets['train'] = ApplyTransform(datasets['train'], transforms.RandomHorizontalFlip())

dataset_sizes = {"train": len(datasets['train']), "val": len(datasets['train'])}

dataloaders = {x: DataLoader(datasets[x], 32, shuffle=True, num_workers=4) for x in ['train', 'val']}
x, y = next(iter(dataloaders['train']))
print(x.shape, y.shape)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=17)
model.to(device)

optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

since = time.time()

best_model_weights = copy.deepcopy(model.state_dict())
best_acc = 0.0

for epoch in range(epochs):
    print('Epoch {}/{}'.format(epoch, epochs - 1))
    print('-' * 10)

    for phase in ['train', 'val']:
        if phase == 'train':
            model.train()

        else:
            model.eval()

        running_loss = 0.0
        running_corrects = 0

        for batch_idx, (inputs, labels) in enumerate(dataloaders[phase]):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            if batch_idx % 30 == 0:
                now = time.time()
                print("epoch:%d, batch:%d, %f" % (epoch, batch_idx, (now - since) / 1000), )

        if phase == 'train':
            scheduler.step()

        epoch_loss = running_loss / dataset_sizes[phase]
        epoch_acc = running_corrects.double() / dataset_sizes[phase]
        # writer.add_graph('epoch loss', epoch_loss, epoch)
        # writer.add_graph('epoch acc', epoch_acc, epoch)

        print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
        f = open('./wow.txt', 'w')
        f.write('{} Loss: {:.4f} Acc: {:.4f}\n'.format(phase, epoch_loss, epoch_acc))
        f.close()

        if phase == 'val' and epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_weights = copy.deepcopy(model.state_dict())

    torch.save(best_model_weights, './weights/best_weights_b5_class_15.pth')
