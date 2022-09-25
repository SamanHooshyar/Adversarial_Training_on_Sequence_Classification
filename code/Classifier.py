#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
from ProxLSTM import ProximalLSTMCell as pro


# In[2]:


class LSTMClassifier(nn.Module):
    def __init__(self, batch_size, output_size, hidden_size, input_size, epsilon):
        super(LSTMClassifier, self).__init__()

        self.output_size = output_size # should be 9
        self.hidden_size = hidden_size  #the dimension of the LSTM output layer
        self.input_size = input_size  # should be 12
        self.epsilon = epsilon
        self.conv = nn.Conv1d(in_channels= self.input_size, out_channels= 64, kernel_size= 3, stride= 3) # feel free to change out_channels, 
        # kernel_size, stride
        self.relu = nn.ReLU()
        self.lstm = nn.LSTMCell(64, hidden_size)
        self.linear = nn.Linear(self.hidden_size, self.output_size)
        self.proxlstm = pro(self.lstm, self.hidden_size, self.epsilon)
        self.dropout = nn.Dropout(p=0.3)
        self.batchnorm = nn.BatchNorm1d(64)
        self.apply_batchnorm = False
        self.apply_dropout = False


    def forward(self, input, r, batch_size, mode='plain'):
        # do the forward pass
        # pay attention to the order of input dimension.
        # input now is of dimension: batch_size * sequence_length * input_size

        x = F.normalize(input)
        x = x.permute(0,2,1)          
        x = self.conv(x)
        if self.apply_batchnorm:
            x = self.batchnorm(x)
        x = self.relu(x)          
        x = x.permute(2,0,1)
        if self.apply_dropout:
            x = self.dropout(x)
            
        self.lstm_input = x
        
        hx = torch.randn(batch_size, self.hidden_size)
        cx = torch.randn(batch_size, self.hidden_size)
        '''need to be implemented'''
        if mode == 'plain':
            # chain up the layers
            
            for i in range(x.size()[0]):
                hx, cx = self.lstm(x[i], (hx, cx))
            x = self.linear(hx)          
            return(x)

        if mode == 'AdvLSTM':
            # chain up the layers
            # different from mode='plain', you need to add r to the forward pass
            # also make sure that the chain allows computing the gradient with respect to the input of LSTM
            
            x = torch.cat((x, self.lstm_input + self.epsilon*r), 1)
            hx = hx.repeat(2,1)
            cx = cx.repeat(2,1)
            for i in range(x.size()[0]):
                hx, cx = self.lstm(x[i], (hx, cx))
            x = self.linear(hx)          
            return(x)
        
        if mode == 'ProxLSTM':
            # chain up layers, but use ProximalLSTMCell here

            G_final = False
            for i in range(x.size()[0]):
                if i == x.size()[0]-1:
                    G_final = True
                hx, cx = self.proxlstm(x[i], hx, cx, G_final)
            x = self.linear(hx)
            return(x)

