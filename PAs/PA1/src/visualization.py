import numpy as np
import open3d as o3d
from typing import List, Optional

class ResultVisualizer:
    def __init__(self, mesh_path: str):
        """初始化可视化器
        Args:
            mesh_path: 网格文件路径
        """
        self.mesh = o3d.io.read_triangle_mesh(mesh_path)
        self.mesh.compute_vertex_normals()
        
    def create_coordinate_frame(self, size: float = 1.0) -> o3d.geometry.TriangleMesh:
        """创建坐标系"""
        return o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
    
    def create_sphere(self, center: np.ndarray, radius: float, color: List[float]) -> o3d.geometry.TriangleMesh:
        """创建单个球体"""
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
        sphere.translate(center)
        sphere.paint_uniform_color(color)
        sphere.compute_vertex_normals()
        return sphere
    
    def visualize(self, 
                 sphere_centers: List[np.ndarray], 
                 radius: float,
                 mesh_alpha: float = 0.5,
                 sphere_colors: Optional[List[List[float]]] = None,
                 window_name: str = "Sphere Placement Result"):
        """可视化网格和球体
        Args:
            sphere_centers: 球心坐标列表
            radius: 球体半径
            mesh_alpha: 网格透明度
            sphere_colors: 球体颜色列表，如果为None则使用默认颜色
            window_name: 窗口名称
        """
        # 创建可视化窗口
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=window_name, width=1280, height=720)
        
        # 设置网格属性
        self.mesh.paint_uniform_color([0.8, 0.8, 0.8])  # 浅灰色
        vis.add_geometry(self.mesh)
        
        # 添加坐标系
        coordinate_frame = self.create_coordinate_frame()
        vis.add_geometry(coordinate_frame)
        
        # 添加球体
        default_colors = [
            [1, 0, 0],  # 红色
            [0, 1, 0],  # 绿色
            [0, 0, 1],  # 蓝色
            [1, 1, 0],  # 黄色
            [1, 0, 1],  # 品红
            [0, 1, 1],  # 青色
        ]
        
        for i, center in enumerate(sphere_centers):
            color = sphere_colors[i] if sphere_colors else default_colors[i % len(default_colors)]
            sphere = self.create_sphere(center, radius, color)
            vis.add_geometry(sphere)
        
        # 设置渲染选项
        opt = vis.get_render_option()
        opt.background_color = np.array([1, 1, 1])  # 白色背景
        opt.mesh_show_wireframe = True
        opt.mesh_show_back_face = True
        opt.point_size = 5.0
        # opt.transparency_mode = o3d.visualization.RenderOption.TransparencyOption.FilterTransparent
        
        # 设置网格透明度
        self.mesh.paint_uniform_color([0.8, 0.8, 0.8])
        vertices_colors = np.asarray(self.mesh.vertex_colors)
        vertices_colors[:, 3] = mesh_alpha
        
        # 设置视角
        ctr = vis.get_view_control()
        ctr.set_zoom(0.8)
        ctr.set_front([1, 1, 1])
        ctr.set_lookat([0, 0, 0])
        ctr.set_up([0, 0, 1])
        
        # 运行可视化
        vis.run()
        vis.destroy_window()
        
    def save_screenshot(self, 
                       sphere_centers: List[np.ndarray], 
                       radius: float,
                       output_path: str,
                       mesh_alpha: float = 0.5,
                       sphere_colors: Optional[List[List[float]]] = None):
        """保存可视化结果为图片
        Args:
            sphere_centers: 球心坐标列表
            radius: 球体半径
            output_path: 输出图片路径
            mesh_alpha: 网格透明度
            sphere_colors: 球体颜色列表
        """
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False)
        
        # 设置场景
        self.mesh.paint_uniform_color([0.8, 0.8, 0.8])
        vis.add_geometry(self.mesh)
        vis.add_geometry(self.create_coordinate_frame())
        
        for i, center in enumerate(sphere_centers):
            color = sphere_colors[i] if sphere_colors else [1, 0, 0]
            sphere = self.create_sphere(center, radius, color)
            vis.add_geometry(sphere)
        
        # 设置渲染选项
        opt = vis.get_render_option()
        opt.mesh_show_wireframe = True
        opt.background_color = np.array([1, 1, 1])
        
        # 保存图片
        vis.capture_screen_image(output_path, True)
        vis.destroy_window()