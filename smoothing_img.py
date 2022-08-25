import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from skimage.io import imread, imshow, imsave
from skimage.exposure import histogram
from scipy.signal import convolve
from scipy.ndimage import median_filter
from pathlib import Path
import os
import glob
import cv2 as cv
from keras.preprocessing import image
from keras.preprocessing.image import save_img

def linear_smoothing_filter(img):
    mask = 1 / 9 * np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
    print('Линейный сглаживающий фильтр')
    new_img = cv.filter2D(img, cv.CV_64F, mask)
    return new_img


def median_filter_(img):
    print('Медианный фильтр')
    new_img = median_filter(img, size=3)
    return new_img

def averaging_filter(img):
    print('Averaging')
    new_img = cv.blur(img, (3, 3))
    return new_img

def gaussian_filter(img):
    print('Gaussian')
    new_img = cv.GaussianBlur(img, (3, 3), 0)
    return new_img


# def prewitt(img):
#     print('оператор Прюит')
#     window1 = 1 / 6 * np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
#     window2 = 1 / 6 * np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
#
#     s1 = cv.filter2D(img, cv.CV_64F, window1)
#     s2 = cv.filter2D(img, cv.CV_64F, window2)
#
#     e = np.sqrt(np.square(s1) + np.square(s2))
#
#     threshold = 25
#     new_img = e > threshold
#     new_img = new_img * 255
#     return new_img


# def laplacian(img):
#     print('Лппласиан')
#     window = 1 / 6 * np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
#
#     s = cv.filter2D(img, cv.CV_64F, window)
#     e = np.abs(s)
#
#     threshold = 10
#     new_img = e > threshold
#     new_img = new_img * 255
#     return new_img


def create_folder_if_not_exists(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    return folder_name


def smoothing(dataset_path, new_folder):
    for img_folder in glob.glob(dataset_path):
        for img_file in glob.glob(img_folder+'/*'):
            class_name = Path(img_folder).name

            img = imread(img_file)
            img_name = Path(img_file).stem

            lin_img = linear_smoothing_filter(img)
            folder_name = f'smoothing/{new_folder}/linear/{class_name}'
            folder_name = create_folder_if_not_exists(folder_name)
            imsave(f'{folder_name}/{img_name}.jpeg', lin_img.astype(np.uint8))

            median_img = median_filter_(img)
            folder_name = f'smoothing/{new_folder}/median/{class_name}'
            folder_name = create_folder_if_not_exists(folder_name)
            imsave(f'{folder_name}/{img_name}.jpeg', median_img.astype(np.uint8))

            av_img = averaging_filter(img)
            folder_name = f'smoothing/{new_folder}/averaging/{class_name}'
            folder_name = create_folder_if_not_exists(folder_name)
            imsave(f'{folder_name}/{img_name}.jpeg', av_img.astype(np.uint8))

            gaussian_img = gaussian_filter(img)
            folder_name = f'smoothing/{new_folder}/gaussian/{class_name}'
            folder_name = create_folder_if_not_exists(folder_name)
            imsave(f'{folder_name}/{img_name}.jpeg', gaussian_img.astype(np.uint8))

            print()



def main():
    original_imgs_path = '../imagenet-mini/val/*'
    new_folder = 'imagenet-mini/'
    smoothing(original_imgs_path, new_folder)
    for model_name in glob.glob('attack_results/*'):
        for attack_name in glob.glob(f'{model_name}/*'):
            adversarial_imgs_path = f'{attack_name}/*'
            new_folder = f'attacked/{Path(model_name).stem}/{Path(attack_name).stem}/'
            smoothing(adversarial_imgs_path, new_folder)


if __name__ == '__main__':
    main()

