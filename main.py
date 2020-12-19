from efficientnet_pytorch import EfficientNet

import os
import torch
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import time
import copy
import numpy

from torch.optim import lr_scheduler
from torchvision import datasets
from efficientnet_pytorch import EfficientNet

'''hyper parameter'''
num_classes = 17
model = EfficientNet.from_pretrained('efficientnet-b3', num_classes=num_classes)
batch_size = 16
epochs = 30

data_dir = '../DL_Final/barkSNU/'

''' data pre-process'''
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((200, 200)),
        transforms.ColorJitter(brightness=(0.7, 1.3)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.Resize((200, 200))
    ])
}

fraction = 0.1


def data_fraction(dataset):
    return torch.utils.data.Subset(dataset,
                                   numpy.random.choice(len(dataset), int(len(dataset) * fraction), replace=False))


image_datasets = {x: data_fraction(datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x]))
                  for x in ['train', 'val']}

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                              shuffle=True, num_workers=4)
               for x in ['train', 'val']}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
# class_names = image_datasets['train'].classes

''' data load '''
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = EfficientNet.from_pretrained('efficientnet-b3', num_classes=15)
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
                print(f'epoch: {epoch}, batch_idx: {batch_idx}, time: {time.time() - since}\n')

        if phase == 'train':
            scheduler.step()

        epoch_loss = running_loss / dataset_sizes[phase]
        epoch_acc = running_corrects.double() / dataset_sizes[phase]
        # writer.add_graph('epoch loss', epoch_loss, epoch)
        # writer.add_graph('epoch acc', epoch_acc, epoch)

        print('{} Loss: {:.4f} Acc: {:.4f}'.format(
            phase, epoch_loss, epoch_acc))

        if phase == 'val' and epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_weights = copy.deepcopy(model.state_dict())
            torch.save(best_model_weights, './weights/best_weights_b5_class_15.pth')
