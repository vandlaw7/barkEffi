from PIL import Image
import numpy as np
import random as rn
import scipy.stats as ss


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


def max_R(img_rgb):  # G가 가장 큰 픽셀 찾기
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


def randomCroppedImage(img_rgb, coverrate, num=5, iteration=10):
    n = 0
    i = 0
    while (n < num and i < iteration):
        i = i + 1
        print("iter : ", i)
        croppedimg = (random_cropping(normalize(img_rgb)))
        img_R = max_R(croppedimg)
        img_G = max_G(croppedimg)
        if cover_rate(img_R) + cover_rate(img_G) < coverrate:
            n = n + 1
            print("R 피복도 : ", cover_rate(img_R))
            print("G 피복도 : ", cover_rate(img_G))
            print("피복도 : ", cover_rate(img_R) + cover_rate(img_G))


def min_B(img_rgb):  # G가 가장 큰 픽셀 찾기
    minB = np.zeros((img_rgb.shape[0], img_rgb.shape[1]))
    for i in range(img_rgb.shape[0]):
        for j in range(img_rgb.shape[1]):
            if (img_rgb[i, j, 2] + 0 < img_rgb[i, j, 0] and img_rgb[i, j, 2] + 0 < img_rgb[i, j, 1]):
                minB[i, j] = 1

    return minB


def normalize_min_B(img_rgb):
    return min_B(normalize(img_rgb))


def std(img_rgb, n=0, k=1):
    output = np.zeros((img_rgb.shape[0], img_rgb.shape[1]))
    img_rgb_rate = ratio_N(img_rgb, n)
    normalized_rate = (img_rgb_rate - np.mean(img_rgb_rate, axis=0)) / np.std(img_rgb_rate, axis=0)
    for i in range(img_rgb.shape[0]):
        for j in range(img_rgb.shape[1]):
            if normalized_rate[i, j] > k:
                output[i, j] = 1

    return output


def ratio_N(img_rgb):  # G/(R + G + B)로 비교하는 함수
    ratios = np.zeros((img_rgb.shape[0], img_rgb.shape[1], 3))
    sum_RGB = np.zeros((img_rgb.shape[0], img_rgb.shape[1]))
    int_img_rgb = makeint(img_rgb)
    for i in range(int_img_rgb.shape[0]):
        for j in range(int_img_rgb.shape[1]):
            for k in range(3):
                sum_RGB[i, j] = sum(int_img_rgb[i, j]) + 1

                ratios[i, j, k] = int_img_rgb[i, j, k] / sum_RGB[i, j]

    return ratios


def makeint(img_rgb):
    flat_img_rgb = np.reshape(img_rgb, (-1))
    intvalue = [float(i) for i in flat_img_rgb]
    intvalue = np.reshape(intvalue, (img_rgb.shape[0], img_rgb.shape[1], img_rgb.shape[2]))

    return intvalue


if __name__ == '__main__':
    filepath = "D:/OneDrive - SNU/SNU/8학기/딥러닝의 기초/프로젝트/테스트용 사진/방해물/메타세콰이아3(37%).jpg"
    img_rgb = np.array(Image.open(filepath))

    randomCroppedImage(img_rgb, 0.1, 5, 25)
