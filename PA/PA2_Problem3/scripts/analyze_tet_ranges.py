import os
import numpy as np
from pathlib import Path


def load_tetrahedral_grid(file_path):
    """
    Load tetrahedral grid data from npz file
    Args:
        file_path: Path to tetrahedral grid file
    Returns:
        vertices: Vertex coordinates [num_vertices, 3]
        tets: Tetrahedron indices [num_tets, 4]
    """
    data = np.load(file_path)
    vertices = data['vertices']
    tets = data['tets']
    return vertices, tets

def analyze_tet_ranges(tet_dir, output_file=None):
    """
    Analyze coordinate ranges of tetrahedral grids
    Args:
        tet_dir: Path to tetrahedral grid directory
        output_file: Path to save analyses (optional)
    """
    npz_files = [f for f in os.listdir(tet_dir) if f.endswith('_compress.npz')]
    analyses = []

    for filename in npz_files:
        file_path = os.path.join(tet_dir, filename)
        vertices, _ = load_tetrahedral_grid(file_path)
        
        min_vals = np.min(vertices, axis=0)
        max_vals = np.max(vertices, axis=0)
        ranges = max_vals - min_vals
        
        analyses.append({
            'filename': filename,
            'min': min_vals,
            'max': max_vals,
            'range': ranges,
            'num_vertices': len(vertices)
        })

    report = ["Tetrahedral Grid Ranges Analysis Report:\n"]
    report.append(f"{'Filename':<20} | {'X Range':<15} | {'Y Range':<15} | {'Z Range':<15} | {'Vertices'}")
    report.append("-" * 85)
    
    for res in analyses:
        line = f"{res['filename']:<20} | {res['range'][0]:<10.4f} ({res['min'][0]:.4f}-{res['max'][0]:.4f}) | " \
               f"{res['range'][1]:<10.4f} ({res['min'][1]:.4f}-{res['max'][1]:.4f}) | " \
               f"{res['range'][2]:<10.4f} ({res['min'][2]:.4f}-{res['max'][2]:.4f}) | {res['num_vertices']}"
        report.append(line)

    final_report = '\n'.join(report)
    if output_file:
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            f.write(final_report)
    else:
        print(final_report)

if __name__ == "__main__":
    config = {
        'tet_dir': 'data/tets',
        'output_file': 'analyses/tet_ranges_report.txt'
    }
    analyze_tet_ranges(**config)