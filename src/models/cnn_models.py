

import torch
import torch.nn as nn
from torch.cuda.amp import autocast
import os
import torch.nn.functional as F


class CNNModel(nn.Module):
    
    
    def __init__(self, label_dim=527, fstride=10, tstride=10, input_fdim=128, input_tdim=1024, imagenet_pretrain=True, audioset_pretrain=False, model_size='base384', verbose=True):

        super(CNNModel, self).__init__()


        self.re_cnn = torch.nn.Conv2d(1, 1, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.re_cnn_2 = torch.nn.Conv2d(1, 1, kernel_size=(3, 3), stride=(1, 1), padding=1)
        
        self.mlp_head = nn.Sequential(nn.Linear(1024*128, label_dim))
        #self.mlp_head_2 = nn.Sequential(nn.LayerNorm(self.original_embedding_dim), nn.Linear(self.original_embedding_dim, label_dim),
          #nn.Linear(label_dim, label_dim),                                
         #nn.Linear(label_dim, label_dim)                              )

        #.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0
        
        
        
        
    def forward(self, x):
        """
        :param x: the input spectrogram, expected shape: (batch_size, time_frame_num, frequency_bins), e.g., (12, 1024, 128)
        :return: prediction
        """
        
        
        
        x = x.unsqueeze(1)
        x = F.relu(self.re_cnn(x))
        x = F.relu(self.re_cnn_2(x))
        #x = x + self.repr.repeat(x.size(dim=0),1,1,1)
        #print(x.size(dim=0))#4
        #print(x.size(dim=1))#1
        #print(x.size(dim=2))#1024
        #print(x.size(dim=3))#128
        #x = x.transpose(2, 3)
        x = torch.flatten(x, 1)
        #torch.flatten(input, start_dim=0, end_dim=- 1)
        

        x = self.mlp_head(x)
        return x

