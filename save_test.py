import torch
from efficientnet_pytorch import EfficientNet
from torch.optim import lr_scheduler

import torch.optim as optim
import torch.nn as nn
import time
import copy




device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=17)
model.to(device)

optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

since = time.time()

best_model_weights = copy.deepcopy(model.state_dict())

torch.save(best_model_weights, './weights/best_weights_b5_class_15.pth')
