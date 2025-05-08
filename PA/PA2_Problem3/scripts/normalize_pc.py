import os
import numpy as np
import trimesh
from pathlib import Path

def normalize_point_clouds(pc_dir, output_dir):
    """
    Normalize point clouds to [-0.5, 0.5] range
    Args:
        pc_dir: Path to input point cloud directory
        output_dir: Path to output normalized point cloud directory
    """
    pc_files = [f for f in os.listdir(pc_dir) if f.endswith('_pts.ply')]
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    for filename in pc_files:
        file_path = os.path.join(pc_dir, filename)
        mesh = trimesh.load(file_path)
        points = mesh.vertices if hasattr(mesh, 'vertices') else np.asarray(mesh.vertices)
        
        min_vals = np.min(points, axis=0)
        max_vals = np.max(points, axis=0)
        center = (min_vals + max_vals) / 2.0
        scale = np.max(max_vals - min_vals)
        
        normalized_points = (points - center) / scale
        
        output_path = os.path.join(output_dir, filename)
        normalized_mesh = trimesh.PointCloud(normalized_points)
        normalized_mesh.export(output_path)

if __name__ == "__main__":
    config = {
        'pc_dir': 'data/pcs',
        'output_dir': 'data/pcs_normalized'
    }
    normalize_point_clouds(**config)