import torch
from torch import nn
import torch.nn.functional as F
import utils
from utils.args import args
import torchaudio.transforms as T

input_size = 1024
hidden_size = 512

class MLP(nn.Module):
    def __init__(self):
        num_classes, valid_labels= utils.utils.get_domains_and_labels(args)
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)
        
        self.dropout= nn.Dropout(args.models.RGB.dropout)
        self.relu = nn.ReLU()
        self.avg_pool = nn.AdaptiveAvgPool1d(1) 
        
    def forward(self, x):
        if args.feat_avg:   #*Feature Averaging
            x = self.avg_pool(x.permute(0, 2, 1))  
            x = x.permute(0, 2, 1)
            x = x.squeeze(dim=1)
            x = self.fc1(x)
            x = self.dropout(x)
            x = self.relu(x)
            x = self.fc2(x)
            x = self.dropout(x)
            x = self.relu(x)
            logits = self.fc3((x))
        else:              #*Logits Averaging
            
            x = self.fc1(x)
            x = self.dropout(x)
            x = self.relu(x)
            x = self.fc2(x)
            x = self.dropout(x)
            x = self.relu(x)
            logits = self.fc3(x)
            logits = self.avg_pool(logits.permute(0, 2, 1)) 
            logits = logits.permute(0, 2, 1)
            logits = logits.squeeze(dim=1)
       
        return logits, {}


class CNN_EMG(nn.Module):
    # Sampling frequency is 160 Hz
    # With 32 samples the frequency resolution after FFT is 160 / 32
    def __init__(self):
        num_classes, valid_labels, source_domain, target_domain = utils.utils.get_domains_and_labels(args)
        super(CNN_EMG, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=16, out_channels=128, kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=2)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(256*1*3, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(p=0.5)
        
    def forward(self, x):
        n_fft = 32
        win_length = None
        hop_length = 16
        
        spectrogram = T.Spectrogram(
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            center=True,
            pad_mode="reflect",
            power=2.0,
            normalized=True
        )
        print(x.shape) # 32,50/100, 16
        x = torch.stack([spectrogram(x[:,:, i]) for i in range(16)], dim=1).float() #dim=2 32,17,16,7 #dim=1 32,16,17,7
        print(x.shape)
        
        x = self.pool(torch.relu(self.conv1(x)))
        print(x.shape) #[32, 128, 8, 3])
        x = self.pool(torch.relu(self.conv2(x)))
        print(x.shape) #[32, 256, 3, 1])
        x = x.view(-1, 256*1*3 )
        print(x.shape)
        x = self.dropout(torch.relu(self.fc1(x)))
        print(x.shape)
        x = self.fc2(x)
        return x, {}