import time
import numpy

import torch
from torch import optim, nn

from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader

from torchvision import datasets, transforms

from efficientnet_pytorch import EfficientNet

effi_version = 0
num_classes = 17

BATCH_SIZE = 32
N_WORKERS = 4
N_EPOCHS = 5

HEIGHT = 137
WIDTH = 236

check_period = 100

INPUT_PATH = '../DL_Final/barkSNU/test'
WEIGHTS_FILE = './weights/best_weights_b5_class_15.pth'

fraction = 0.1


def data_fraction(dataset, fraction=fraction):
    return torch.utils.data.Subset(dataset,
                                   numpy.random.choice(len(dataset), int(len(dataset) * fraction), replace=False))


test_transform = transforms.Compose([
    transforms.Resize((200, 200)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

test_image_dataset = data_fraction(datasets.ImageFolder(INPUT_PATH, test_transform))
test_set_size = len(test_image_dataset)
max_batch_idx = test_set_size // BATCH_SIZE

test_image_loaded = DataLoader(test_image_dataset, batch_size=BATCH_SIZE,
                               shuffle=False, num_workers=4)

device = torch.device("cuda:0")
model = EfficientNet.from_pretrained(f'efficientnet-b{effi_version}', num_classes=num_classes)
model.load_state_dict(torch.load(WEIGHTS_FILE))
model.to(device)

criterion = nn.CrossEntropyLoss()

''' for train, not here'''
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)


def convert_to_preferred_format(sec):
    sec = sec % (24 * 3600)
    hour = sec // 3600
    sec %= 3600
    min = sec // 60
    sec %= 60
    return "%02d:%02d:%02d" % (hour, min, sec)


''' inference run'''
since = time.time()
running_loss = 0.0
running_corrects = 0
for batch_idx, (inputs, labels) in enumerate(test_image_loaded):
    inputs = inputs.to(device)
    labels = labels.to(device)

    with torch.set_grad_enabled(False):
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)

    running_loss += loss.item() * inputs.size(0)
    running_corrects += torch.sum(preds == labels.data)

    if batch_idx % check_period == 0:
        print(f'batch_idx: {batch_idx} / {max_batch_idx}, '
              f'time: {convert_to_preferred_format(time.time() - since)}')

total_loss = running_loss / test_set_size
total_acc = running_corrects.double() / test_set_size

f = open('./test_result.txt', 'a')
print(f'total loss: {total_loss}, total accuracy: {total_acc}')
f.write(f'total loss: {total_loss}, total accuracy: {total_acc}\n')
f.close()