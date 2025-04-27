import os
import numpy as np
import trimesh
from pathlib import Path

def analyze_point_cloud_ranges(pc_dir, output_file=None):
    """
    Analyze 3D coordinate ranges of all PLY files in the point cloud directory
    Args:
        pc_dir: Path to point cloud directory
        output_file: Path to save analyses (optional)
    """
    pc_files = [f for f in os.listdir(pc_dir) if f.endswith('_pts.ply')]
    analyses = []

    for filename in pc_files:
        file_path = os.path.join(pc_dir, filename)
        mesh = trimesh.load(file_path)
        
        # Get point cloud data (handling both mesh and point cloud cases)
        points = mesh.vertices if hasattr(mesh, 'vertices') else np.asarray(mesh.vertices)
        
        # Calculate coordinate ranges
        min_vals = np.min(points, axis=0)
        max_vals = np.max(points, axis=0)
        ranges = max_vals - min_vals
        
        analyses.append({
            'filename': filename,
            'min': min_vals,
            'max': max_vals,
            'range': ranges
        })

    # Print analyses
    report = ["Point Cloud Ranges Analysis Report:\n"]
    report.append(f"{'Filename':<20} | {'X Range':<15} | {'Y Range':<15} | {'Z Range':<15}")
    report.append("-" * 70)
    
    for res in analyses:
        line = f"{res['filename']:<20} | {res['range'][0]:<10.4f} ({res['min'][0]:.4f}-{res['max'][0]:.4f}) | " \
               f"{res['range'][1]:<10.4f} ({res['min'][1]:.4f}-{res['max'][1]:.4f}) | " \
               f"{res['range'][2]:<10.4f} ({res['min'][2]:.4f}-{res['max'][2]:.4f})"
        report.append(line)

    # Output to file or console
    final_report = '\n'.join(report)
    if output_file:
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            f.write(final_report)
    else:
        print(final_report)

if __name__ == "__main__":
    # Use path settings from configuration
    config = {
        'pc_dir': 'data/pcs',        # Corresponds to pc_dir in config.yaml
        'output_file': 'analyses/pc_ranges_report.txt'
    }
    analyze_point_cloud_ranges(**config)