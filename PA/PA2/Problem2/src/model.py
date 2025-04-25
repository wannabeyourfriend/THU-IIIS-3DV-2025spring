import torch
import torch.nn as nn
import torch.nn.functional as F

class PointCloudAutoEncoder(nn.Module):
    def __init__(self, num_points=1024, feature_dim=512):
        """
        :param num_points: 
        :param feature_dim:                     
        """
        super(PointCloudAutoEncoder, self).__init__()
        
        self.encoder = nn.Sequential(
            # input: [batch_size, 3, 224, 224]
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            # Res Block 1
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # Res Block 2
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            # Res Block 3
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            
            # Global Average Pooling
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Linear(512, feature_dim)
        
        self.decoder = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, num_points * 3)  
        )
        
        self.num_points = num_points
        
    def forward(self, x):
        """
        :param x: Input Image [batch_size, 3, H, W]
        :return: Predicted Point Cloud [batch_size, num_points, 3]
        """
        x = self.encoder(x)
        x = x.view(x.size(0), -1)  
        x = self.fc(x)
        x = self.decoder(x)
        x = x.view(-1, self.num_points, 3) 
        
        return x
