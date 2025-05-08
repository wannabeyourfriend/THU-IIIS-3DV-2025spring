import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class PositionalEncoding(nn.Module):
    """
    Implements Fourier feature mapping to transform 3D coordinates into higher-dimensional space
    Similar to the positional encoding method in NeRF
    """
    def __init__(self, num_freqs=10, include_input=True):
        super(PositionalEncoding, self).__init__()
        self.num_freqs = num_freqs
        self.include_input = include_input
        self.funcs = [torch.sin, torch.cos]
        
        self.out_dim = 0
        if self.include_input:
            self.out_dim += 3
        self.out_dim += 2 * 3 * self.num_freqs
    
    def forward(self, x):
        """
        Args:
            x: input coordinates [batch_size, ..., 3]
        Returns:
            encoded: encoded features [batch_size, ..., out_dim]
        """
        orig_shape = list(x.shape)
        x = x.reshape(-1, 3)
        
        encoded = []
        if self.include_input:
            encoded.append(x)
            
        for freq_idx in range(self.num_freqs):
            freq = 2.0 ** freq_idx
            for func in self.funcs:
                encoded.append(func(x * freq * np.pi))
        
        encoded = torch.cat(encoded, dim=-1)
        encoded = encoded.reshape(orig_shape[:-1] + [self.out_dim])
        
        return encoded

class MLPNetwork(nn.Module):
    """
    MLP network for predicting SDF values and deformation vectors
    """
    def __init__(self, pos_enc_freqs=10, hidden_dim=256, num_layers=8):
        super(MLPNetwork, self).__init__()
        
        self.pos_encoder = PositionalEncoding(num_freqs=pos_enc_freqs)
        input_dim = self.pos_encoder.out_dim
        
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        
        for _ in range(num_layers - 2):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        
        self.sdf_layer = nn.Linear(hidden_dim, 1)
        self.deform_layer = nn.Linear(hidden_dim, 3)
        
        for layer in self.layers:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
        
        nn.init.xavier_uniform_(self.sdf_layer.weight, gain=0.1)
        nn.init.zeros_(self.sdf_layer.bias)
        nn.init.xavier_uniform_(self.deform_layer.weight, gain=0.1)
        nn.init.zeros_(self.deform_layer.bias)
        
    def forward(self, x):
        """
        Args:
            x: input vertex coordinates [batch_size, num_vertices, 3]
        Returns:
            deform: predicted deformation vectors [batch_size, num_vertices, 3]
            sdf: predicted SDF values [batch_size, num_vertices, 1]
        """
        x = self.pos_encoder(x)
        
        feat = x
        for i, layer in enumerate(self.layers):
            feat_new = layer(feat)
            if i < len(self.layers) - 1:
                feat_new = F.leaky_relu(feat_new, negative_slope=0.1)
            feat = feat_new
        
        sdf = self.sdf_layer(feat)
        deform = 0.1 * torch.tanh(self.deform_layer(feat))
        
        return deform, sdf

class Conv3DNetwork(nn.Module):
    """
    3D convolutional network for predicting SDF values and deformation vectors
    """
    def __init__(self, grid_size=64, pos_enc_freqs=10, hidden_dim=64, num_conv_layers=4):
        super(Conv3DNetwork, self).__init__()
        
        self.grid_size = grid_size
        self.pos_encoder = PositionalEncoding(num_freqs=pos_enc_freqs)
        input_dim = self.pos_encoder.out_dim
        
        self.init_feature_dim = hidden_dim
        
        self.input_layer = nn.Linear(input_dim, self.init_feature_dim)
        
        self.conv_layers = nn.ModuleList()
        for i in range(num_conv_layers):
            in_channels = self.init_feature_dim
            out_channels = self.init_feature_dim
            
            self.conv_layers.append(
                nn.Sequential(
                    nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
                    nn.BatchNorm3d(out_channels),
                    nn.ReLU(inplace=True)
                )
            )
            
            if i < num_conv_layers - 1:
                self.conv_layers.append(
                    nn.Sequential(
                        nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
                        nn.BatchNorm3d(out_channels),
                        nn.ReLU(inplace=True)
                    )
                )
        
        self.sdf_layer = nn.Conv3d(self.init_feature_dim, 1, kernel_size=1)
        self.deform_layer = nn.Conv3d(self.init_feature_dim, 3, kernel_size=1)
        
    def forward(self, x):
        """
        Args:
            x: input vertex coordinates [batch_size, num_vertices, 3]
        Returns:
            deform: predicted deformation vectors [batch_size, num_vertices, 3]
            sdf: predicted SDF values [batch_size, num_vertices, 1]
        """
        if len(x.shape) == 2:
            x = torch.unsqueeze(x, 0)
        batch_size, num_vertices, _ = x.shape
        
        x_encoded = self.pos_encoder(x)
        
        features = self.input_layer(x_encoded)
        
        coords = (x + 1) / 2
        grid_coords = (coords * (self.grid_size - 1)).long()
        
        volume_features = torch.zeros(
            (batch_size, self.init_feature_dim, self.grid_size, self.grid_size, self.grid_size),
            device=x.device
        )
        
        for b in range(batch_size):
            for v in range(num_vertices):
                i, j, k = grid_coords[b, v]
                volume_features[b, :, i, j, k] = features[b, v]
        
        for conv_layer in self.conv_layers:
            volume_features = conv_layer(volume_features)
        
        volume_sdf = self.sdf_layer(volume_features)
        volume_deform = self.deform_layer(volume_features)
        
        sdf = torch.zeros((batch_size, num_vertices, 1), device=x.device)
        deform = torch.zeros((batch_size, num_vertices, 3), device=x.device)
        
        for b in range(batch_size):
            for v in range(num_vertices):
                i, j, k = grid_coords[b, v]
                sdf[b, v, 0] = volume_sdf[b, 0, i, j, k]
                deform[b, v] = volume_deform[b, :, i, j, k]
        
        return deform, sdf

class DMTetModel(nn.Module):
    """
    DMTet model integrating network and Marching Tetrahedra algorithm
    """
    def __init__(self, network_type='mlp', **kwargs):
        super(DMTetModel, self).__init__()
        
        if network_type.lower() == 'mlp':
            self.network = MLPNetwork(**kwargs)
        elif network_type.lower() == 'conv3d':
            self.network = Conv3DNetwork(**kwargs)
        else:
            raise ValueError(f"Unknown network type: {network_type}")
    
    def forward(self, vertices, tets=None):
        """
        Args:
            vertices: tetrahedral mesh vertices [batch_size, num_vertices, 3]
            tets: tetrahedral indices [batch_size, num_tets, 4]
        Returns:
            sdf: predicted SDF values [batch_size, num_vertices, 1]
            deform: predicted deformation vectors [batch_size, num_vertices, 3]
            deformed_vertices: deformed vertices [batch_size, num_vertices, 3]
        """
        deform, sdf = self.network(vertices)
        deformed_vertices = vertices + deform
        
        return sdf, deform, deformed_vertices