import numpy as np
import open3d as o3d
import trimesh
from itertools import combinations

class VolumeCoverage:
    def __init__(self, mesh_path: str):
        self.mesh = o3d.io.read_triangle_mesh(mesh_path)
        self.trimesh = trimesh.load(mesh_path)
        self.mesh_volume = abs(self.trimesh.volume)
        
    def sphere_volume(self, radius):
        """计算单个球体的体积"""
        return (4/3) * np.pi * radius**3
    
    def sphere_intersection_volume(self, center1, center2, radius):
        """计算两个球体相交的体积
        Args:
            center1, center2: 两个球心的坐标
            radius: 球体半径
        """
        d = np.linalg.norm(center1 - center2)
        if d >= 2 * radius:
            return 0
        
        h = radius - d**2 / (4 * radius)
        return (2 * np.pi * radius**2 * h) / 3
    
    def is_sphere_inside_mesh(self, center, radius):
        """检查球体是否完全在mesh内部
        Args:
            center: 球心坐标
            radius: 球体半径
        """
        # 在球体表面采样点进行检查
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
        sphere.translate(center)
        points = np.asarray(sphere.sample_points_uniformly(number_of_points=100).points)
        
        # 检查所有采样点是否在mesh内部
        return all(self.mesh.is_inside_points(points))
    
    def sphere_intersection_points(self, centers, radius, num_points=1000):
        """计算多个球体相交区域的采样点
        Args:
            centers: 球心坐标数组，shape为(n, 3)
            radius: 球体半径
            num_points: 采样点数量
        Returns:
            相交区域的采样点
        """
        # 生成包围盒内的随机点
        bbox_min = np.min(centers - radius, axis=0)
        bbox_max = np.max(centers + radius, axis=0)
        points = np.random.uniform(bbox_min, bbox_max, (num_points, 3))
        
        # 检查点是否在所有球体内部
        in_all_spheres = np.ones(num_points, dtype=bool)
        for center in centers:
            distances = np.linalg.norm(points - center, axis=1)
            in_all_spheres &= (distances <= radius)
        
        return points[in_all_spheres]

    def multi_sphere_intersection_volume(self, centers, radius):
        """计算多个球体相交的体积
        Args:
            centers: 球心坐标数组，shape为(n, 3)
            radius: 球体半径
        Returns:
            相交体积
        """
        if len(centers) <= 1:
            return 0
        elif len(centers) == 2:
            return self.sphere_intersection_volume(centers[0], centers[1], radius)
        
        # 使用蒙特卡洛方法估算多球相交体积
        intersection_points = self.sphere_intersection_points(centers, radius)
        if len(intersection_points) == 0:
            return 0
        
        # 计算包围盒体积
        bbox_min = np.min(centers - radius, axis=0)
        bbox_max = np.max(centers + radius, axis=0)
        bbox_volume = np.prod(bbox_max - bbox_min)
        
        # 估算相交体积
        return bbox_volume * len(intersection_points) / 1000

    def calculate_coverage(self, sphere_centers, radius):
        """计算球体覆盖的体积及其占mesh体积的百分比
        Args:
            sphere_centers: 球心坐标列表，shape为(n, 3)
            radius: 球体半径
        Returns:
            coverage_volume: 覆盖的体积
            coverage_percentage: 覆盖百分比
        """
        n = len(sphere_centers)
        if n == 0:
            return 0, 0
        
        # 计算单个球体体积之和
        total_volume = n * self.sphere_volume(radius)
        
        # 使用容斥原理计算重叠部分
        for k in range(2, n + 1):
            sign = (-1)**(k-1)
            for combo in combinations(range(n), k):
                centers = np.array([sphere_centers[i] for i in combo])
                intersection_volume = self.multi_sphere_intersection_volume(centers, radius)
                total_volume += sign * intersection_volume
        
        coverage_percentage = (total_volume / self.mesh_volume) * 100
        return total_volume, coverage_percentage