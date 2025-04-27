import os
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import glob
import random
import open3d as o3d

class CubeDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None, num_points=1024, clean=True):
        """
        :param root_dir: Dataset root directory
        :param split: Dataset split ('train', 'val', 'test')
        :param transform: Image transformations
        :param num_points: Number of points in point cloud
        :param clean: Whether to use clean point cloud data
        """
        self.root_dir = os.path.join(root_dir, 'cubes', 'clean' if clean else 'noisy')
        print(f"Loading dataset from: {self.root_dir}")
        
        if not os.path.exists(self.root_dir):
            raise ValueError(f"Dataset directory does not exist: {self.root_dir}")
        
        self.transform = transform
        self.num_points = num_points
        self.clean = clean
        
        # Get all existing cube directories
        cube_dirs = []
        for i in range(100):
            dir_path = os.path.join(self.root_dir, str(i))
            if os.path.exists(dir_path):
                cube_dirs.append(str(i))
        
        if not cube_dirs:
            raise ValueError(f"No cube directories found in {self.root_dir}")
        
        cube_dirs = sorted(cube_dirs)
        print(f"Found {len(cube_dirs)} cube directories")
        
        if split == 'train':
            self.cube_dirs = cube_dirs[:70]  # 70% for training
        elif split == 'val':
            self.cube_dirs = cube_dirs[70:85]  # 15% for validation
        else:  # test
            self.cube_dirs = cube_dirs[85:]  # 15% for testing
        
        self.samples = []
        
        for cube_dir in self.cube_dirs:
            cube_path = os.path.join(self.root_dir, cube_dir)
            image_files = [os.path.join(cube_path, f"{i}.png") for i in range(16)]
            point_files = [os.path.join(cube_path, f"{i}.ply") for i in range(16)]
            
            for img_file, pc_file in zip(image_files, point_files):
                if os.path.exists(img_file) and os.path.exists(pc_file):
                    self.samples.append((img_file, pc_file))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, pc_path = self.samples[idx]
        
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        pcd = o3d.io.read_point_cloud(pc_path)
        point_cloud = np.asarray(pcd.points)
        
        if point_cloud.shape[0] > self.num_points:
            indices = np.random.choice(point_cloud.shape[0], self.num_points, replace=False)
            point_cloud = point_cloud[indices]
        elif point_cloud.shape[0] < self.num_points:
            indices = np.random.choice(point_cloud.shape[0], self.num_points, replace=True)
            point_cloud = point_cloud[indices]
        
        point_cloud = torch.from_numpy(point_cloud).float()
        
        return image, point_cloud
