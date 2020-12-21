import time
import numpy
import numpy as np

import torch
from torch import optim, nn

from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader

from torchvision import datasets, transforms

from efficientnet_pytorch import EfficientNet

import random as rn
import scipy.stats as ss

import sys
import getopt

GREEN_ESCAPE = False
INPUT_PATH = '../DL_Final/barkSNU/test'


def main(argv):
    global INPUT_PATH, GREEN_ESCAPE
    try:
        opts, etc_args = getopt.getopt(argv[1:], "dt")
    except getopt.GetoptError:
        print("error")
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-d':
            print("green dataset selection")
            INPUT_PATH = '../DL_Final/barkSNU/green_hinder'
        elif opt == "-t":
            print("green escape transform")
            GREEN_ESCAPE = True

    return


if __name__ == '__main__':
    main(sys.argv)

print(f'GREEN_ESCAPE: {GREEN_ESCAPE}, INPUT_PATH: {INPUT_PATH}')

effi_version = 0
num_classes = 17

BATCH_SIZE = 64
N_WORKERS = 4
N_EPOCHS = 5

WEIGHTS_FILE = './weights/best_weights_b0_class_17_aug_50.pth'

fraction = 1
check_period = 1

'''crop selection to escape green hinder'''
GREEN_GAP = 5
GREEN_THRESHOLD = 0.1
GREEN_ITERATION_MAX = 20


def how_much_green_dominated(image, gap=GREEN_GAP):
    w, h = image.size
    green_win = 0
    rgb_im = image.convert('RGB')
    for i in range(w):
        for j in range(h):
            r, g, b = rgb_im.getpixel((i, j))
            if g > r + gap and g > b + gap:
                green_win += 1

    return green_win / (w * h)


def is_green_dominated(image, gap=GREEN_GAP, threshold=GREEN_THRESHOLD):
    return how_much_green_dominated(image, gap) > threshold


class RandomCropMy(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, image, gap=GREEN_GAP, threshold=GREEN_THRESHOLD,
                 iteration_max=GREEN_ITERATION_MAX):
        w, h = image.size
        new_w, new_h = self.output_size

        iteration = 0
        best_image = image
        green_best = 1
        while iteration < iteration_max:
            left = np.random.randint(0, w - new_w)
            upper = np.random.randint(0, h - new_h)

            crop_image = image.crop((left, upper, left + new_w, upper + new_h))
            green_record = how_much_green_dominated(crop_image, gap)
            if green_record > threshold:
                iteration += 1
                if green_record < green_best:
                    green_best = green_record
                    best_image = crop_image
            else:
                return crop_image

        return best_image


def normalize(img_rgb):
    return ss.zscore(img_rgb)


def random_cropping(img_rgb):  # 랜덤 크롭, 임의의 점을 찍고 그 점을 기준으로 +- 165만큼을 고름
    x = rn.randrange(165, img_rgb.shape[0] - 165)
    y = rn.randrange(165, img_rgb.shape[1] - 165)
    cropped = img_rgb[x - 165:x + 166, y - 165:y + 166, :]
    return cropped


def max_G(img_rgb):  # G가 가장 큰 픽셀 찾기
    maxG = np.zeros((img_rgb.shape[0], img_rgb.shape[1]))
    for i in range(img_rgb.shape[0]):
        for j in range(img_rgb.shape[1]):
            if (img_rgb[i, j, 1] > img_rgb[i, j, 0] + 0.15 and img_rgb[i, j, 1] > img_rgb[i, j, 2] + 0.15):
                maxG[i, j] = 1

    return maxG


def max_R(img_rgb):  # R이 가장 큰 픽셀 찾기
    maxR = np.zeros((img_rgb.shape[0], img_rgb.shape[1]))
    for i in range(img_rgb.shape[0]):
        for j in range(img_rgb.shape[1]):
            if (img_rgb[i, j, 0] > img_rgb[i, j, 1] + 0.15 and img_rgb[i, j, 0] > img_rgb[i, j, 2] + 0.15):
                maxR[i, j] = 1

    return maxR


def cover_rate(img_rgb):  # 피복도 구하기 - 픽셀 중 분류된 값의 비 구함
    nonzero = 0
    for i in range(img_rgb.shape[0]):
        for j in range(img_rgb.shape[1]):
            if img_rgb[i, j] != 0:
                nonzero = nonzero + 1
    covered = nonzero / (img_rgb.shape[0] * img_rgb.shape[1])

    return covered


class RandomCropMy2(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, image, gap=GREEN_GAP, threshold=GREEN_THRESHOLD,
                 iteration_max=GREEN_ITERATION_MAX):
        w, h = image.size
        new_w, new_h = self.output_size

        img_rgb = np.array(image)
        image_normalized = normalize(img_rgb)

        iteration = 0
        best_image = image
        green_best = 1
        while iteration < iteration_max:
            croppedimg = random_cropping(image_normalized)
            img_R = max_R(croppedimg)
            img_G = max_G(croppedimg)
            cover_rate_rg = cover_rate(img_R) + cover_rate(img_G)

            if cover_rate_rg > threshold:
                iteration += 1
                if green_record < green_best:
                    green_best = green_record
                    best_image = crop_image
            else:
                return croppedimg

        return best_image


def data_fraction(dataset, fraction=fraction):
    return torch.utils.data.Subset(dataset,
                                   numpy.random.choice(len(dataset), int(len(dataset) * fraction), replace=False))


def convert_to_preferred_format(sec):
    sec = sec % (24 * 3600)
    hour = sec // 3600
    sec %= 3600
    min = sec // 60
    sec %= 60
    return "%02d:%02d:%02d" % (hour, min, sec)


test_transform = transforms.Compose([
    transforms.Resize((200, 200)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

new_transform = transforms.Compose([
    RandomCropMy(200),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

for i in range(10):
    print(f'{i}th check')
    test_image_dataset = data_fraction(
        datasets.ImageFolder(INPUT_PATH, (new_transform if GREEN_ESCAPE else test_transform)))
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

    f = open('./test_result2.txt', 'a')
    print(f'total loss: {total_loss}, total accuracy: {total_acc}')
    f.write(
        f'GREEN_ESCAPE: {GREEN_ESCAPE}, INPUT_PATH: {INPUT_PATH} total loss: {total_loss}, total accuracy: {total_acc}\n')
    f.close()
