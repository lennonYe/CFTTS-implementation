import torch
import torch.nn as nn



class CausalConv1d(nn.Module):
    def __init__(self,in_channels, out_channels,kernel_size,stride=1,padding=0,dilation=1):
        super(CausalConv1d,self).__init__()
        self.conv1d = nn.Conv1d(in_channels,out_channels,kernel_size,stride=stride,padding=2*padding*dilation,dilation=dilation)
    
    def forward(self,input):
        (batch_size,ndim,length) = input.size()
        conv_out = self.conv1d(input)
        return conv_out[:,:,0:length]


import torch
import torch.nn.functional as F
import torch.nn as nn

class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1):
        super(CausalConv1d, self).__init__()
        self.conv1d = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=2*padding*dilation, dilation=dilation)
    
    def forward(self, input):
        (batch_size, ndim, length) = input.size()
        conv_out = self.conv1d(input)
        return conv_out[:, :, :length]

class CPCCodeNet(nn.Module):
    def __init__(self):
        super(CPCCodeNet, self).__init__()
        self.downsampling_net = nn.Sequential(*[
            CausalConv1d(80, 128, 3, padding=1),
            nn.ReLU(),
            CausalConv1d(128, 128, 3, padding=1),
            nn.ReLU(),
            #nn.MaxPool1d(2, stride=1, padding=0), 
            CausalConv1d(128, 128, 3, padding=1),
            nn.ReLU(),
            #nn.MaxPool1d(2, stride=1, padding=0), 
            CausalConv1d(128, 128, 3, padding=1),
            nn.ReLU(),
            #nn.MaxPool1d(2, stride=2, padding=0),
            CausalConv1d(128, 128, 3, padding=1)
        ])
        self.cpc_lstm = nn.LSTM(128, 512, batch_first=True, bidirectional=False, num_layers=2)
        self.cpc_lowrank_fc = nn.Linear(512, 64)
        self.cpc_anchor_net = nn.Sequential(*[
            CausalConv1d(128, 128, 3, padding=1, dilation=1),
            nn.ReLU(),
            CausalConv1d(128, 128, 3, padding=1, dilation=2),
            nn.ReLU(),
            CausalConv1d(128, 128, 3, padding=1, dilation=4),
            nn.ReLU(),
            CausalConv1d(128, 128, 3, padding=1, dilation=8),
            nn.ReLU()
        ])
    
    def forward(self, mel_cep, reverse_idx):
        (batch_size, length, ndim) = mel_cep.size()
        
        # 对 mel_cep 进行下采样
        fwd_downsample_out = self.downsampling_net(mel_cep.transpose(1, 2))
        
        # 生成 CPCCode
        cpc_code, _ = self.cpc_lstm(fwd_downsample_out.transpose(1, 2))
        cpc_code_lowrank = self.cpc_lowrank_fc(cpc_code)
        
        # reverse mel_cep & downsampling
        reverse_mel_cep = torch.gather(mel_cep, 1, reverse_idx.unsqueeze(2).expand(-1, -1, ndim))
        reverse_downsample_out = self.downsampling_net(reverse_mel_cep.transpose(1, 2))
        
        # generate CPCAnchor
        cpc_anchor = self.cpc_anchor_net(reverse_downsample_out)
        
        # align anchor's timestep and code
        cpc_anchor = torch.gather(cpc_anchor, 2, reverse_idx.unsqueeze(1).expand(-1, cpc_code_lowrank.size(-1), -1))
        
        return cpc_code_lowrank, cpc_anchor.transpose(1, 2)

    def extract(self, mel_cep):
        fwd_downsample_out = self.downsampling_net(mel_cep.transpose(1, 2))
        cpc_code, _ = self.cpc_lstm(fwd_downsample_out.transpose(1, 2))
        cpc_code_lowrank = self.cpc_lowrank_fc(cpc_code)
        return cpc_code_lowrank


