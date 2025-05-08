import os
import torch
import numpy as np
import open3d as o3d
from torch.utils.data import Dataset, DataLoader


class PointCloudDataset(Dataset):
    """
    点云数据集类，用于加载和预处理点云数据
    """
    def __init__(self, data_dir, point_files=None, normalize=True):
        """
        初始化点云数据集
        Args:
            data_dir: 点云数据目录
            point_files: 指定的点云文件列表，如果为None则加载目录中所有.ply文件
            normalize: 是否对点云进行归一化
        """
        self.data_dir = data_dir
        self.normalize = normalize
        
        if point_files is None:
            # 获取所有.ply文件
            self.point_files = [f for f in os.listdir(data_dir) if f.endswith('_pts.ply')]
        else:
            self.point_files = point_files
    
    def __len__(self):
        return len(self.point_files)
    
    def __getitem__(self, idx):
        file_path = os.path.join(self.data_dir, self.point_files[idx])
        
        # 使用Open3D加载点云
        pcd = o3d.io.read_point_cloud(file_path)
        points = np.asarray(pcd.points).astype(np.float32)
        
        # 归一化点云到[-1, 1]范围
        if self.normalize:
            center = np.mean(points, axis=0)
            points = points - center
            scale = np.max(np.abs(points))
            points = points / scale
        else:
            center = np.zeros(3)
            scale = 1.0
        
        return {
            'points': torch.from_numpy(points),
            'file_name': self.point_files[idx],
            'center': center,
            'scale': scale
        }


class TetrahedralGridLoader:
    """
    四面体网格加载器，用于加载和预处理四面体网格数据
    """
    def __init__(self, tet_dir):
        """
        初始化四面体网格加载器
        Args:
            tet_dir: 四面体网格数据目录
        """
        self.tet_dir = tet_dir
        self.available_grids = self._get_available_grids()
    
    def _get_available_grids(self):
        """
        获取可用的四面体网格分辨率
        Returns:
            available_grids: 可用的网格分辨率列表
        """
        grid_files = [f for f in os.listdir(self.tet_dir) if f.endswith('_compress.npz')]
        available_grids = [int(f.split('_')[0]) for f in grid_files]
        return sorted(available_grids)
    
    def load_grid(self, grid_res, device=None):
        """
        加载指定分辨率的四面体网格
        Args:
            grid_res: 网格分辨率
            device: 设备，如果为None则返回CPU张量
        Returns:
            vertices: 顶点坐标 [num_vertices, 3]
            tets: 四面体索引 [num_tets, 4]
        """
        if grid_res not in self.available_grids:
            raise ValueError(f"网格分辨率 {grid_res} 不可用，可用的分辨率: {self.available_grids}")
        
        file_path = os.path.join(self.tet_dir, f"{grid_res}_compress.npz")
        data = np.load(file_path)
        
        vertices = torch.from_numpy(data['vertices']).float()
        tets = torch.from_numpy(data['tets']).long()
        
        if device is not None:
            vertices = vertices.to(device)
            tets = tets.to(device)
        
        return vertices, tets


def create_dataloader(data_dir, batch_size=1, shuffle=True, num_workers=0):
    """
    创建点云数据加载器
    Args:
        data_dir: 点云数据目录
        batch_size: 批量大小
        shuffle: 是否打乱数据
        num_workers: 数据加载的工作线程数
    Returns:
        dataloader: 数据加载器
    """
    dataset = PointCloudDataset(data_dir)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )
    return dataloader