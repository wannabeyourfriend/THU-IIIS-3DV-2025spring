import numpy as np
import open3d as o3d
import trimesh
from typing import List, Tuple
from .sample import MeshSampler
from .volume_coverage import VolumeCoverage

class GreedySpheresPlacer:
    def __init__(self, mesh_path: str, radius: float):
        self.mesh_path = mesh_path
        self.sampler = MeshSampler(mesh_path)
        self.coverage_calculator = VolumeCoverage(mesh_path)
        self.radius = radius
        self.placed_centers = []
        self.trimesh = trimesh.load(mesh_path)
        
    def is_sphere_inside_mesh(self, center: np.ndarray) -> bool:
        """使用 trimesh 检查球体是否在网格内部"""
        # 在球体表面采样点
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=self.radius)
        sphere.translate(center)
        points = np.asarray(sphere.sample_points_uniformly(number_of_points=100).points)
        
        # 使用 trimesh 检查点是否在网格内部
        return all(self.trimesh.contains(points))
        
    def evaluate_position(self, center: np.ndarray) -> float:
        """评估在某个位置放置球体的收益"""
        # 计算加入新球体后的总覆盖体积
        test_centers = self.placed_centers + [center]
        new_volume, _ = self.coverage_calculator.calculate_coverage(test_centers, self.radius)
        
        # 计算当前覆盖体积
        if not self.placed_centers:
            current_volume = 0
        else:
            current_volume, _ = self.coverage_calculator.calculate_coverage(
                self.placed_centers, self.radius
            )
            
        # 返回增加的体积
        return new_volume - current_volume
    
    def find_best_position(self, candidate_centers: np.ndarray) -> np.ndarray:
        """在候选位置中找到最佳放置位置"""
        max_gain = -float('inf')
        best_center = None
        
        for center in candidate_centers:
            gain = self.evaluate_position(center)
            if gain > max_gain:
                max_gain = gain
                best_center = center
                
        return best_center
    
    def place_spheres(self, max_spheres: int, num_candidates: int = 1000) -> Tuple[List[np.ndarray], float]:
        """贪心放置球体
        Args:
            max_spheres: 最大球体数量
            num_candidates: 每次迭代的候选点数量
        Returns:
            placed_centers: 放置的球心位置列表
            coverage_ratio: 最终的覆盖率
        """
        for _ in range(max_spheres):
            # 采样候选点
            candidates = self.sampler.sample_candidate_centers(num_candidates, self.radius)
            
            if len(candidates) == 0:
                break
                
            # 找到最佳位置
            best_center = self.find_best_position(candidates)
            if best_center is None:
                break
                
            # 放置球体
            self.placed_centers.append(best_center)
            
        # 计算最终覆盖率
        _, coverage_ratio = self.coverage_calculator.calculate_coverage(
            self.placed_centers, self.radius
        )
        
        return self.placed_centers, coverage_ratio
    
    def visualize_result(self):
        """可视化放置结果"""
        # 创建可视化窗口
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        
        # 添加网格
        mesh = o3d.io.read_triangle_mesh(self.sampler.mesh_path)
        mesh.compute_vertex_normals()
        vis.add_geometry(mesh)
        
        # 添加球体
        for center in self.placed_centers:
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=self.radius)
            sphere.translate(center)
            sphere.paint_uniform_color([1, 0, 0])  # 红色球体
            vis.add_geometry(sphere)
        
        # 设置视角和渲染
        vis.get_render_option().mesh_show_wireframe = True
        vis.get_render_option().line_width = 2.0
        vis.run()
        vis.destroy_window()