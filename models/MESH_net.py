import torch
from torch import nn
import torch.nn.functional as F
import utils
from utils.args import args
import torchaudio.transforms as T

class EdgeConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EdgeConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        # x is the edge features tensor
        return self.conv(x)

class MeshCNN(nn.Module):
    def __init__(self):
        num_classes, valid_labels = utils.utils.get_domains_and_labels(args)
        super(MeshCNN, self).__init__()
        self.edge_conv1 = EdgeConv(3, 64)
        self.edge_conv2 = EdgeConv(64, 128)
        self.edge_conv3 = EdgeConv(128, 256)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, edge_features):
        x = self.edge_conv1(edge_features)
        x = self.edge_conv2(x)
        x = self.edge_conv3(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x