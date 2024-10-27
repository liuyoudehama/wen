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
class twoConvBlock(nn.Module):
  def __init__(self, input_channels, output_channels):
    super(twoConvBlock, self).__init__()
    self.double_conv = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=3, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_channels, output_channels, kernel_size=3, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True)
        )

  def forward(self, input_image):
    return self.double_conv(input_image)
    

class downStep(nn.Module):
  def __init__(self):
    super(downStep, self).__init__()
    self.max_pool_layer = nn.MaxPool2d((2, 2), stride=(2, 2))

  def forward(self, input_feature_maps):
    return self.max_pool_layer(input_feature_maps)

class upStep(nn.Module):
  def __init__(self, input_channels, output_channels):
    super(upStep, self).__init__()
    self.up_sampling_layer = nn.ConvTranspose2d(input_channels, output_channels, kernel_size=2, stride=2)
    self.conv = twoConvBlock(input_channels, output_channels)
        
  def forward(self, down_input, left_input):
    up_sampled = self.up_sampling_layer(down_input)

    larger_width = left_input.size()[-1]
    larger_height = left_input.size()[-2]
    smaller_width = up_sampled.size()[-1]
    smaller_height = up_sampled.size()[-2]
    diff_width = larger_width - smaller_width
    diff_height = larger_height - smaller_height
    left_diff = diff_width // 2
    top_diff = diff_height // 2
    
    # should not use padding, should crop..
    # pad_up_sampled = F.pad(up_sampled, [diff_w // 2, diff_w - diff_w // 2, diff_h // 2, diff_h - diff_h // 2])
    left_input = left_input[..., top_diff: top_diff + smaller_height, left_diff: left_diff + smaller_width]

    mixed = torch.cat([left_input, up_sampled], dim=1)
    return mixed
    
class UNet(nn.Module):
  def __init__(self):
    super(UNet, self).__init__()
    self.conv1 = twoConvBlock(1,64)
    self.conv2 = twoConvBlock(64, 128)
    self.conv3 = twoConvBlock(128, 256)
    self.conv4 = twoConvBlock(256, 512)
    self.conv5 = twoConvBlock(512, 1024)
    
    self.conv6 = twoConvBlock(1024, 512)
    self.conv7 = twoConvBlock(512, 256)
    self.conv8 = twoConvBlock(256, 128)
    self.conv9 = twoConvBlock(128, 64)
    
    self.conv10 = nn.Conv2d(in_channels=64, out_channels=2, kernel_size=1, stride=1)

    self.down_step = downStep()

    self.upstep1 = upStep(1024, 512)
    self.upstep2 = upStep(512, 256)
    self.upstep3 = upStep(256, 128)
    self.upstep4 = upStep(128, 64)
    
  def forward(self, input_img):
    conv1_result = self.conv1(input_img)
    # print(f"conv1_result shape: {conv1_result.shape}")
      
    downsampled = self.down_step(conv1_result)
    conv2_result = self.conv2(downsampled)
    # print(f"conv2_result shape: {conv2_result.shape}")
      
    downsampled = self.down_step(conv2_result)
    conv3_result = self.conv3(downsampled)
    # print(f"conv3_result shape: {conv3_result.shape}")
    
    downsampled = self.down_step(conv3_result)
    conv4_result = self.conv4(downsampled)
    # print(f"conv4_result shape: {conv3_result.shape}")
    
    downsampled = self.down_step(conv4_result)
    down_input1 = self.conv5(downsampled)
    # print(f"down_input1 shape: {down_input1.shape}")
    
    mixed = self.upstep1(down_input1, conv4_result)
    down_input2 = self.conv6(mixed)
    # print(f"down_input2 shape: {down_input2.shape}")
    
    mixed = self.upstep2(down_input2, conv3_result)
    down_input3 = self.conv7(mixed)
    # print(f"down_input3 shape: {down_input3.shape}")
      
    mixed = self.upstep3(down_input3, conv2_result)
    down_input4 = self.conv8(mixed)
    # print(f"down_input4 shape: {down_input4.shape}")
          
    mixed = self.upstep4(down_input4, conv1_result)
    # print(f"mixed shape: {mixed.shape}")
    
    out = self.conv9(mixed)
    # print(f"out shape: {out.shape}")
      
    out = self.conv10(out)
    # print(f"out shape: {out.shape}")

    # larger_width = input_img.size()[-1]
    # larger_height = input_img.size()[-2]
    # smaller_width = out.size()[-1]
    # smaller_height = out.size()[-2]
    # diff_width = larger_width - smaller_width
    # diff_height = larger_height - smaller_height
    
    # out = F.pad(out, [diff_width // 2, diff_width - diff_width // 2, diff_height // 2, diff_height - diff_height // 2])
    return out