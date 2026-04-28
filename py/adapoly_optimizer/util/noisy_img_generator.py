import os
    
from collections import OrderedDict
import sys
from unicodedata import decimal
import pandas as pd



sys.path.append("/home/LAB/lufh/ada_optimizer/py")
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
# import cv2 as cv
from torch import nn

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

class NoisyImageGenerator():

    def __init__(self, feature_num=3, size=(1,3,32,32)):
        self.features = []
        for i in range(feature_num):
            self.features.append(torch.rand(size))


    def generate_image_w_noise(self, feat_index:int, path, size=(1000,3,32,32), strength=0.05):
        labels = []
        imgs = []
        train_val = []
        images = self.features[feat_index] + (strength * torch.randn(size))
        try:
            os.makedirs(path)
            print(f"Folders '{path}' created successfully.")
        except FileExistsError:
                print(f"Folders '{path}' already exist.")
        for i in range(size[0]):
            to_pil_image = transforms.ToPILImage()
            pil_image = to_pil_image(images[i])
            image_path = path+('/%d.png'%i)
            pil_image.save(image_path)
            labels.append(feat_index)
            imgs.append(image_path)
            if i < 0.8 * size[0]:
                train_val.append(0)
            else:
                train_val.append(1)
        return labels, imgs, train_val





account = '/lichenghao/huY'
if __name__ == '__main__':
    # generator = NoisyImageGenerator(feature_num=4)
    # img_items = []
    # total_targets=[]
    # total_imgs=[]
    # train_vals=[]
    # targets,img_names,train_val = generator.generate_image_w_noise(0,account + '/ada_optimizer/assets/custdataset1/imagewsmallnoise', size=(1000,3,32,32), strength=0.1)
    # total_targets += targets
    # total_imgs += img_names
    # train_vals += train_val
    # targets,img_names,train_val = generator.generate_image_w_noise(1,account + '/ada_optimizer/assets/custdataset1/diffimagewlargenoise', size=(1000,3,32,32), strength=0.5)
    # total_targets += targets
    # total_imgs += img_names
    # train_vals += train_val
    # targets,img_names,train_val = generator.generate_image_w_noise(0,account + '/ada_optimizer/assets/custdataset1/imagewlargenoise', size=(200,3,32,32), strength=0.5)
    # total_targets += targets
    # total_imgs += img_names
    # train_vals += train_val
    # targets,img_names,train_val = generator.generate_image_w_noise(2,account + '/ada_optimizer/assets/custdataset1/fewimagewsmallnoise', size=(200,3,32,32), strength=0.1)
    # total_targets += targets
    # total_imgs += img_names
    # train_vals += train_val
    # targets,img_names,train_val = generator.generate_image_w_noise(3,account + '/ada_optimizer/assets/custdataset1/fewimagewlargenoise', size=(200,3,32,32), strength=0.5)
    # total_targets += targets
    # total_imgs += img_names
    # train_vals += train_val

    generator = NoisyImageGenerator(feature_num=5)
    img_items = []
    total_targets=[]
    total_imgs=[]
    train_vals=[]
    targets,img_names,train_val = generator.generate_image_w_noise(0,account + '/ada_optimizer/assets/custdataset1/imagewsmallnoise1', size=(1000,3,32,32), strength=0.1)
    total_targets += targets
    total_imgs += img_names
    train_vals += train_val
    targets,img_names,train_val = generator.generate_image_w_noise(1,account + '/ada_optimizer/assets/custdataset1/imagewsmallnoise2', size=(1000,3,32,32), strength=0.1)
    total_targets += targets
    total_imgs += img_names
    train_vals += train_val
    targets,img_names,train_val = generator.generate_image_w_noise(2,account + '/ada_optimizer/assets/custdataset1/imagewsmallnoise3', size=(1000,3,32,32), strength=0.1)
    total_targets += targets
    total_imgs += img_names
    train_vals += train_val
    targets,img_names,train_val = generator.generate_image_w_noise(3,account + '/ada_optimizer/assets/custdataset1/imagewsmallnoise4', size=(1000,3,32,32), strength=0.1)
    total_targets += targets
    total_imgs += img_names
    train_vals += train_val
    targets,img_names,train_val = generator.generate_image_w_noise(4,account + '/ada_optimizer/assets/custdataset1/imagewsmallnoise5', size=(1000,3,32,32), strength=0.1)
    total_targets += targets
    total_imgs += img_names
    train_vals += train_val
    
    img_items.append(total_imgs)
    img_items.append(total_targets)
    img_items.append(train_vals)
    pd.DataFrame(data=img_items).transpose().to_csv(account + '/ada_optimizer/assets/custdataset1/target.csv')


   
