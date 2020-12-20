# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 16:51:35 2020

@author: young
"""

import matplotlib.pylab as plt
import matplotlib.image as mpimg
import numpy as np
import random as rn


def random_cropping(img_rgb):  # 랜덤 크롭, 임의의 점을 찍고 그 점을 기준으로 +- 165만큼을 고름
    x = rn.randrange(165, img_rgb.shape[0] - 165)
    y = rn.randrange(165, img_rgb.shape[1] - 165)
    cropped = img_rgb[x - 165:x + 166, y - 165:y + 166, :]
    return cropped


def randomCroppedImage(img_rgb, coverrate, num=5, iteration=10):
    n = 0
    i = 0
    while (n < num and i < iteration):
        i = i + 1
        print("iter : ", i)
        croppedimg = random_cropping(img_rgb)
        img = max_G(croppedimg)
        if cover_rate(img) < coverrate:
            n = n + 1
            plt.figure(figsize=(20, 5))
            plt.subplot(141)
            plt.imshow(croppedimg)
            plt.subplot(142)
            plt.imshow(img)
            plt.show()
            print("피복도 : ", cover_rate(img))


def cover_rate(img_rgb):  # 피복도 구하기 - 픽셀 중 분류된 값의 비 구함
    nonzero = 0
    for i in range(img_rgb.shape[0]):
        for j in range(img_rgb.shape[1]):
            if img_rgb[i, j] != 0:
                nonzero = nonzero + 1
    covered = nonzero / (img_rgb.shape[0] * img_rgb.shape[1])

    return covered


def max_G(img_rgb):  # G가 가장 큰 픽셀 찾기
    maxG = np.zeros((img_rgb.shape[0], img_rgb.shape[1]))
    for i in range(img_rgb.shape[0]):
        for j in range(img_rgb.shape[1]):
            if (img_rgb[i, j, 1] > img_rgb[i, j, 0] and img_rgb[i, j, 1] > img_rgb[i, j, 2] + 10):
                maxG[i, j] = 1

    return maxG


def sort_RGB(img_rgb, n):  # R 값으로 비교하는 함수 n : 0 - R, 1- G, 2 - B
    sort = np.zeros((img_rgb.shape[0], img_rgb.shape[1]))
    if n >= img_rgb.shape[2]:
        return "invalid"
    for i in range(img_rgb.shape[0]):
        for j in range(img_rgb.shape[1]):
            if img_rgb[i, j, n] <= 150:
                sort[i, j] = 0
            else:
                sort[i, j] = 1  # img_rgb[i, j]  # 1

    return sort


def ratio_G(img_rgb):  # G/(R + G + B)로 비교하는 함수
    ratios = np.zeros((img_rgb.shape[0], img_rgb.shape[1]))
    for i in range(img_rgb.shape[0]):
        for j in range(img_rgb.shape[1]):

            ratios[i, j] = img_rgb[i, j, 1] / sum(img_rgb[i, j])
            if ratios[i, j] <= 0.34:
                ratios[i, j] = 0

    return ratios


def extract_RGB(img_rgb, n):  # RGB 값 리스트 n : 0 - R, 1- G, 2 - B
    RGB = []
    if n >= img_rgb.shape[2]:
        return "invalid"
    for i in range(img_rgb.shape[0]):
        for j in range(img_rgb.shape[1]):
            RGB.append(img_rgb[i, j, n])

    return RGB


if __name__ == '__main__':
    filepath = "D:/OneDrive - SNU/SNU/8학기/딥러닝의 기초/프로젝트/테스트용 사진/나뭇잎/상수리나무_나뭇잎/IMG_6416(23%).jpg"
    img_rgb = mpimg.imread(filepath)

    randomCroppedImage(img_rgb, 0.5, 5, 15)
