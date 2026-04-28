import glob
import os

import cv2
import pandas as pd
import torch
from PIL import Image
from torchvision import utils as vutils


def read_csv_path(path):
    """
    read csv file.
    :param path: where the file is.
    :return: return the result of reading file.
    """
    return pd.read_csv(path)


def read_csv(path, file_name):
    """
    read csv file.
    :param path: where the file is.
    :param file_name: the file name.
    :return: return the result of reading file.
    """
    return pd.read_csv(path + (r'/%s.csv' % file_name))


def save_csv_file(dataset, path, file_name, header=True, index=False):
    """
    save data into a file
    :param dataset: the data needed to save.
    :param path: where you want to save the file.
    :param file_name: the file name.
    :param header: whether saving the header.
    :param index: whether saving the index.
    :return: whether it is success.
    """
    if not os.path.exists(path):
        os.makedirs(path)
    # header = None代表没有表头
    dataset.to_csv(path + (r'/%s.csv' % file_name), header=header, index=index)


def tif2png_pil(original_path, saved_path):
    """
    transform tif file into png via PIL package.
    :param original_path: the path of tif file.
    :param saved_path: the saved path of png file.
    :return: none
    """
    counts = 0
    files = os.listdir(original_path)
    for file in files:
        try:
            if file.endswith('tif'):
                tif_file = os.path.join(original_path, file)

                file = file[:-3] + 'png'
                png_file = os.path.join(saved_path, file)
                im = Image.open(tif_file)
                im.save(png_file)
                print(png_file)
                counts += 1
        finally:
            print('fail')

    print('%d done' % counts)


def tif2png_cv(original_path, saved_path):
    """
    transform tif file into png via CV package.
    :param original_path: the path of tif file.
    :param saved_path: the saved path of png file.
    :return: none
    """
    counts = 0
    files = os.listdir(original_path)
    for file in files:
        try:
            if file.endswith('tif'):
                tif_file = os.path.join(original_path, file)

                file = file[:-3] + 'png'
                png_file = os.path.join(saved_path, file)
                im = cv2.imread(tif_file)
                cv2.imwrite(png_file, im)
                print(png_file)
                counts += 1
        finally:
            print('fail')
    print('%d done' % counts)


def print_pic_shape(path):
    pictures = glob.glob(path)
    for i in range(len(pictures)):
        img = Image.open(pictures)
        print(img.size)


def pic_crop(pic_path, crop_horizon, crop_vertical, save_path):
    pic_crop(pic_path, crop_horizon, crop_vertical, crop_horizon, crop_vertical, save_path)


def pic_crop(pic_path, crop_left, crop_top, crop_right, crop_bottom, save_path):
    pic_path = os.path.join(pic_path, '*')
    pictures = glob.glob(pic_path)
    for i in range(len(pictures)):
        try:
            img_name = pictures[i].split('\\')[1]
            img = Image.open(pictures[i].replace('\\', '/'))
        except BaseException:
            continue
        size = img.size
        cropped_img = img.crop((crop_left, crop_top, size[0]-crop_right, size[1]-crop_bottom))  # (left, upper, right, lower)
        cropped_img.save(os.path.join(save_path, img_name))


def save_image_tensor(input_tensor: torch.Tensor, filename):
    """
    将tensor保存为图片
    :param input_tensor: 要保存的tensor
    :param filename: 保存的文件名
    """
    # assert (len(input_tensor.shape) == 4 and input_tensor.shape[0] == 1)
    # 复制一份
    input_tensor = input_tensor.clone().detach()
    # 到cpu
    input_tensor = input_tensor.to(torch.device('cpu'))
    # 反归一化
    # input_tensor = unnormalize(input_tensor)
    vutils.save_image(input_tensor, filename)

