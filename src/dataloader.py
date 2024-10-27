import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

import torch.optim as optim

import os

from PIL import Image, ImageOps

import random

#import any other libraries you need below this line

class Cell_data(Dataset):
  def __init__(self, data_dir, size, train = 'True', train_test_split = 0.8, augment_data = True):
    ##########################inputs##################################
    #data_dir(string) - directory of the data#########################
    #size(int) - size of the images you want to use###################
    #train(boolean) - train data or test data#########################
    #train_test_split(float) - the portion of the data for training###
    #augment_data(boolean) - use data augmentation or not#############
    
    super(Cell_data, self).__init__()
    self.data_dir = data_dir
    self.image_root_path = os.path.join(self.data_dir, "scans")
    self.label_root_path = os.path.join(self.data_dir, "labels")
    self.image_file_names = os.listdir(self.image_root_path)
    self.label_file_names = os.listdir(self.label_root_path)
    
    self.images_path = [os.path.join(self.image_root_path, img) for img in self.image_file_names]
    self.lebels_path = [os.path.join(self.label_root_path, label) for label in self.label_file_names]

    self.size = size
    self.train = train

    split = int(len(self.image_file_names) * train_test_split)
    if self.train:
      self.images_path = self.images_path[:split]
      self.lebels_path = self.lebels_path[:split]
    else:
      self.images_path = self.images_path[split:]
      self.lebels_path = self.lebels_path[split:]
    
    self.train_test_split = train_test_split
    self.augment_data = augment_data

    self.zoom_factor_lower_bound = 0.60
    self.zoom_factor_upper_bound = 0.80
    assert(self.zoom_factor_upper_bound > self.zoom_factor_lower_bound)
    assert(self.zoom_factor_lower_bound > 0 and self.zoom_factor_lower_bound < 1)
    assert(self.zoom_factor_upper_bound > 0 and self.zoom_factor_upper_bound < 1)

    self.rotate_angle_lower_bound = -45
    self.rotate_angle_upper_bound = 45
    
    self.transform = transforms.Compose([
                          transforms.ToTensor(),
                          transforms.Resize((int(self.size), int(self.size)))
                          ])


  def __getitem__(self, idx):
      
      image = self.transform(Image.open(self.images_path[idx]))
      label = self.transform(Image.open(self.lebels_path[idx]))
      
      #data augmentation part
      if self.train and self.augment_data:
        augment_mode = np.random.randint(0, 4)
        if augment_mode == 0:
            #flip image vertically
            image = torch.flip(image, [1])
            label = torch.flip(label, [1])
        elif augment_mode == 1:
            #flip image horizontally
            image = torch.flip(image, [2])
            label = torch.flip(label, [2])
        elif augment_mode == 2:
            #zoom image
            factor = random.random() * (self.zoom_factor_upper_bound - self.zoom_factor_lower_bound) + self.zoom_factor_lower_bound
            transform_in = transforms.CenterCrop((int(factor * self.size), int(factor * self.size)))
            transform_out = transforms.Resize((self.size, self.size))
            image = transform_out(transform_in(image))
            label = transform_out(transform_in(label))
        else:
            #rotate image
            angle = random.randint(self.rotate_angle_lower_bound, self.rotate_angle_upper_bound)
            image = transforms.functional.rotate(image, angle)
            label = transforms.functional.rotate(label, angle)

      label[label != 0] = 1.0
      #return image and mask in tensors
      return image, label
      
      
  def __len__(self):
    return len(self.images_path)