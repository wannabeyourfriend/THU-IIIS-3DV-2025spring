import numpy as np
import open3d as o3d
import trimesh
from typing import List, Tuple

class MeshSampler:
    def __init__(self, mesh_path: str):
        self.mesh_path = mesh_path
        self.mesh = o3d.io.read_triangle_mesh(mesh_path)
        self.mesh.compute_vertex_normals()
        self.trimesh = trimesh.load(mesh_path)
    
    def sample_surface_points(self, num_points: int) -> np.ndarray:
        """在网格表面采样点"""
        points, _ = trimesh.sample.sample_surface(self.trimesh, num_points)
        return points
    
    def generate_internal_points(self, num_points: int, surface_points: np.ndarray) -> np.ndarray:
        """生成网格内部的随机点"""
        bbox_min = np.min(surface_points, axis=0)
        bbox_max = np.max(surface_points, axis=0)
        
        points = []
        while len(points) < num_points:
            # 在包围盒内生成随机点
            candidates = np.random.uniform(bbox_min, bbox_max, (num_points * 2, 3))
            # 使用trimesh检查点是否在网格内部
            inside = self.trimesh.contains(candidates)
            inside_points = candidates[inside]
            points.extend(inside_points[:num_points - len(points)])
            
        return np.array(points[:num_points])
    
    def filter_valid_sphere_centers(self, points: np.ndarray, radius: float) -> np.ndarray:
        """过滤出球心位置合法的点"""
        valid_centers = []
        
        for center in points:
            # 在球体表面采样点进行检查
            sphere = trimesh.primitives.Sphere(radius=radius, center=center)
            surface_points = sphere.sample(100)
            
            # 使用trimesh检查所有采样点是否在网格内部
            if all(self.trimesh.contains(surface_points)):
                valid_centers.append(center)
                
        return np.array(valid_centers)
    
    def sample_candidate_centers(self, num_candidates: int, radius: float) -> np.ndarray:
        """采样候选球心点
        Args:
            num_candidates: 期望的候选点数量
            radius: 球体半径
        Returns:
            有效的球心坐标数组
        """
        # 首先在表面采样点以确定包围盒
        surface_points = self.sample_surface_points(num_candidates * 2)
        
        # 生成内部随机点
        internal_points = self.generate_internal_points(num_candidates * 3, surface_points)
        
        # 过滤出有效的球心点
        valid_centers = self.filter_valid_sphere_centers(internal_points, radius)
        
        # 如果有效点太多，随机选择所需数量
        if len(valid_centers) > num_candidates:
            indices = np.random.choice(len(valid_centers), num_candidates, replace=False)
            valid_centers = valid_centers[indices]
            
        return valid_centers