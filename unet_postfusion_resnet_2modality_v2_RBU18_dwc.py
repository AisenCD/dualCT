# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import init
from torchvision import models
import math

use_relu = False
use_bn = True


#%% bigmb github version
class conv_block(nn.Module):
    """
    Convolution Block 
    """
    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):

        x = self.conv(x)
        return x


class up_conv(nn.Module):
    """
    Up Convolution Block
    """
    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x




class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.depth_conv = nn.Conv2d(in_channels=inplanes,
                                    out_channels=planes,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1,
                                    groups=planes)
        self.point_conv = nn.Conv2d(in_channels=planes,
                                    out_channels=planes,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0,
                                    groups=1)

    def forward(self, x):
        residual = x

        out = self.depth_conv(x)
        out = self.point_conv(out)

        out = out + residual

        return out





# blocks 就是 layers
def _make_layer(block, inplanes, planes, blocks, stride=1):
        layers = []
        layers.append(nn.Conv2d(inplanes, planes, kernel_size=3, stride=1, padding=1,
                                bias=False))

        scale = calculate_scale(layers[-1].weight.data)
        torch.nn.init.uniform_(layers[-1].weight.data, -scale, scale)

        if layers[-1].bias is not None:
            layers[-1].bias.data.zero_()
        
        layers.append(nn.BatchNorm2d(planes))
        
        layers.append(nn.PReLU(planes))
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2, padding=0))

        inplanes = planes
        for i in range(0, blocks):
            layers.append(block(inplanes, planes))

        return nn.Sequential(*layers)


# blocks 就是 layers
def _make_layer_bridge(block, inplanes, planes, blocks, stride=1):
        layers = []
        layers.append(nn.Conv2d(inplanes, planes, kernel_size=3, stride=1, padding=1,
                                bias=False))

        scale = calculate_scale(layers[-1].weight.data)
        torch.nn.init.uniform_(layers[-1].weight.data, -scale, scale)

        if layers[-1].bias is not None:
            layers[-1].bias.data.zero_()
        
        layers.append(nn.BatchNorm2d(planes))
        
        layers.append(nn.PReLU(planes))
        # layers.append(nn.MaxPool2d(kernel_size=2, stride=2, padding=0))

        inplanes = planes
        for i in range(0, blocks):
            layers.append(block(inplanes, planes))

        return nn.Sequential(*layers)


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    global use_bn
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=not use_bn)


def calculate_scale(data):
    if data.dim() == 2:
        scale = math.sqrt(3 / data.size(1))
    else:
        scale = math.sqrt(3 / (data.size(1) * data.size(2) * data.size(3)))
    return scale



class AggBlock(nn.Module):
    def __init__(self, layer_lvl, layers, agg_type='concat', channel_attention=False):
        super(AggBlock, self).__init__()
        self.layer_lvl = layer_lvl
        self.agg_type = agg_type
        self.channel_attention = channel_attention

        if self.layer_lvl == 1:
            inplanes = 64
            outplanes = 128
        elif self.layer_lvl == 2:
            inplanes = 128
            outplanes = 256
        elif self.layer_lvl == 3:
            inplanes = 256
            outplanes = 512
        elif self.layer_lvl == 4:
            inplanes = 512
            outplanes = 1024

        self.agg_layer = _make_layer(BasicBlock, inplanes, outplanes, layers, stride=2)

        if self.agg_type == 'concat':
            self.conv1 = nn.Conv2d(inplanes * 2, inplanes, kernel_size=1)

    def forward(self, prev_x, rgb_x, depth_x):
        if self.agg_type == 'concat':
            x = torch.cat((rgb_x,depth_x), dim=1)
            x = nn.functional.relu(self.conv1(x))

        if self.layer_lvl in [2,3,4]:
            x = prev_x + x

        x = self.agg_layer(x)

        return x




class AggBlock_Bridge(nn.Module):
    def __init__(self, layers, agg_type='concat', channel_attention=False):
        super(AggBlock_Bridge, self).__init__()
        self.agg_type = agg_type
        self.channel_attention = channel_attention

        inplanes = 1024
        outplanes = 1024

        self.agg_layer = _make_layer_bridge(BasicBlock, inplanes, outplanes, layers, stride=2)

        if self.agg_type == 'concat':
            self.conv1 = nn.Conv2d(inplanes, inplanes, kernel_size=1)

    def forward(self, prev_x, x):
        if self.agg_type == 'concat':
            x = nn.functional.relu(self.conv1(x))

        x = prev_x + x
        x = self.agg_layer(x)

        return x




class unet_postfusion_resnet_2modality_v2_rbu18_dwc(nn.Module):
    """
    UNet - Basic Implementation
    Paper : https://arxiv.org/abs/1505.04597
    # https://github.com/bigmb/Unet-Segmentation-Pytorch-Nest-of-Unets/blob/master/Models.py
    """
    def __init__(self, in_ch=4, out_ch=2):
        super(unet_postfusion_resnet_2modality_v2_rbu18_dwc, self).__init__()

        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        # ----- max_pooling -----
        self.pool_1_0 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool_1_1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.pool_1_2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.pool_1_3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.pool_2_0 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool_2_1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.pool_2_2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.pool_2_3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.pool_3_0 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool_3_1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.pool_3_2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.pool_3_3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.pool_4_0 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool_4_1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.pool_4_2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.pool_4_3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # ----- conv_block_down -----
        self.down_1_0 = conv_block(1, filters[0])
        self.down_1_1 = conv_block(1, filters[0])
        # self.down_1_2 = conv_block(1, filters[0])
        # self.down_1_3 = conv_block(1, filters[0])

        self.down_2_0 = conv_block(filters[0], filters[1])
        self.down_2_1 = conv_block(filters[0], filters[1])
        # self.down_2_2 = conv_block(filters[0], filters[1])
        # self.down_2_3 = conv_block(filters[0], filters[1])        

        self.down_3_0 = conv_block(filters[1], filters[2])
        self.down_3_1 = conv_block(filters[1], filters[2])
        # self.down_3_2 = conv_block(filters[1], filters[2])
        # self.down_3_3 = conv_block(filters[1], filters[2])        

        self.down_4_0 = conv_block(filters[2], filters[3])
        self.down_4_1 = conv_block(filters[2], filters[3])
        # self.down_4_2 = conv_block(filters[2], filters[3])
        # self.down_4_3 = conv_block(filters[2], filters[3])

        # ----- bridge -----
        self.bridge = conv_block(filters[3]*2, filters[4]) # channel num to correct
        
        # ----- up_conv -----
        self.up_conv_1 = up_conv(filters[4], filters[3])
        self.up_conv_2 = up_conv(filters[3], filters[2])
        self.up_conv_3 = up_conv(filters[2], filters[1])
        self.up_conv_4 = up_conv(filters[1], filters[0])

        # ----- conv_block_up -----
        self.up_1 = conv_block(filters[3]*3, filters[3])
        self.up_2 = conv_block(filters[2]*3, filters[2])
        self.up_3 = conv_block(filters[1]*3, filters[1])
        self.up_4 = conv_block(filters[0]*3, filters[0])        


        self.agg_layer1 = AggBlock(layer_lvl=1, layers=1, agg_type='concat')
        self.agg_layer2 = AggBlock(layer_lvl=2, layers=2, agg_type='concat')
        self.agg_layer3 = AggBlock(layer_lvl=3, layers=2, agg_type='concat')
        self.agg_layer4 = AggBlock(layer_lvl=4, layers=2, agg_type='concat')
        self.agg_layer_bridge = AggBlock_Bridge(layers=2, agg_type='concat')

        self.out = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):

        # ############################# #
        # ~~~~~~ Encoding path ~~~~~~~  #

        i0 = x[:,0:1,:,:]
        i1 = x[:,1:2,:,:]
        # i2 = x[:,2:3,:,:]
        # i3 = x[:,3:4,:,:]        

        # -----  First Level --------
        down_1_0 = self.down_1_0(i0) 
        down_1_1 = self.down_1_1(i1) 
        # down_1_2 = self.down_1_2(i2) 
        # down_1_3 = self.down_1_3(i3)
        
        
        # -----  Second Level --------
        input_2nd_0 = self.pool_1_0(down_1_0)
        input_2nd_1 = self.pool_1_1(down_1_1)
        # input_2nd_2 = self.pool_1_2(down_1_2)
        # input_2nd_3 = self.pool_1_3(down_1_3)

        down_2_0 = self.down_2_0(input_2nd_0)
        down_2_1 = self.down_2_1(input_2nd_1)
        # down_2_2 = self.down_2_2(input_2nd_2)
        # down_2_3 = self.down_2_3(input_2nd_3)        


        # -----  Third Level --------
        input_3rd_0 = self.pool_2_0(down_2_0)
        input_3rd_1 = self.pool_2_1(down_2_1)
        # input_3rd_2 = self.pool_2_2(down_2_2)
        # input_3rd_3 = self.pool_2_3(down_2_3) 

        down_3_0 = self.down_3_0(input_3rd_0)
        down_3_1 = self.down_3_1(input_3rd_1)
        # down_3_2 = self.down_3_2(input_3rd_2)
        # down_3_3 = self.down_3_3(input_3rd_3)        

        
        # -----  Fourth Level --------
        input_4th_0 = self.pool_3_0(down_3_0)
        input_4th_1 = self.pool_3_1(down_3_1)
        # input_4th_2 = self.pool_3_2(down_3_2)
        # input_4th_3 = self.pool_3_3(down_3_3)         

        down_4_0 = self.down_4_0(input_4th_0)
        down_4_1 = self.down_4_1(input_4th_1)
        # down_4_2 = self.down_4_2(input_4th_2)
        # down_4_3 = self.down_4_3(input_4th_3)

        
        #----- Bridge -----
        down_4_0m = self.pool_4_0(down_4_0)
        down_4_1m = self.pool_4_1(down_4_1)
        # down_4_2m = self.pool_4_2(down_4_2)
        # down_4_3m = self.pool_4_3(down_4_3)
        
        inputBridge = torch.cat((down_4_0m,down_4_1m),dim=1)
        bridge = self.bridge(inputBridge)

        agg_layer1 = self.agg_layer1(None, down_1_0, down_1_1)
        agg_layer2 = self.agg_layer2(agg_layer1, down_2_0, down_2_1)
        agg_layer3 = self.agg_layer3(agg_layer2, down_3_0, down_3_1)
        agg_layer4 = self.agg_layer4(agg_layer3, down_4_0, down_4_1)
        agg_layer5 = self.agg_layer_bridge(bridge, agg_layer4)
        
        # print(bridge.shape)
        # print(agg_layer5.shape)

        bridge = bridge+agg_layer5

        
        
        # ############################# #
        # ~~~~~~ Decoding path ~~~~~~~  #

        up_conv_1 = self.up_conv_1(bridge)
        skip_1 =  torch.cat((up_conv_1, down_4_0, down_4_1),dim=1)
        up_1 = self.up_1(skip_1)

        up_conv_2 = self.up_conv_2(up_1)
        skip_2 = torch.cat((up_conv_2, down_3_0, down_3_1),dim=1) 
        up_2 = self.up_2(skip_2)

        up_conv_3 = self.up_conv_3(up_2)
        skip_3 = torch.cat((up_conv_3, down_2_0, down_2_1),dim=1)  
        up_3 = self.up_3(skip_3)

        up_conv_4 = self.up_conv_4(up_3)
        skip_4 = torch.cat((up_conv_4, down_1_0, down_1_1),dim=1) 
        up_4 = self.up_4(skip_4)

        return self.out(up_4) 
        

