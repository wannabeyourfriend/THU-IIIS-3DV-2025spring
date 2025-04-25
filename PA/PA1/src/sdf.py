import open3d as o3d
import numpy as np
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
import trimesh
import pyvista as pv
def estimate_volume_sdf(mesh_path, num_samples):
    mesh = pv.read(mesh_path)
    start_time = time.time()
    bounds = mesh.bounds
    bbox_min = np.array([bounds[0], bounds[2], bounds[4]])
    bbox_max = np.array([bounds[1], bounds[3], bounds[5]])
    bbox_size = bbox_max - bbox_min
    bbox_volume = np.prod(bbox_size)
    points = np.random.uniform(
        bbox_min, 
        bbox_max, 
        size=(num_samples, 3)
    )
    point_cloud = pv.PolyData(points)
    point_cloud = point_cloud.compute_implicit_distance(mesh)
    signed_distance = point_cloud.get_array('implicit_distance') 
    points_inside = np.sum(signed_distance < 0)
    volume = (points_inside / num_samples) * bbox_volume
    time_cost = time.time() - start_time
    
    return volume, time_cost

def main():
    bunny_path = "objs_approx/bunny.obj"
    tri_mesh = trimesh.load(bunny_path)
    tri_mesh.fix_normals()
    true_volume = tri_mesh.volume
    print(f"True Volume: {true_volume}")
    sample_sizes = [1000, 5000, 10000, 50000, 100000, 500000]
    volumes = []
    times = []
    relative_errors = []
    
    print("\nSDF sampling results:")
    print("Samples | Volume | Error(%) | Time(s)")
    print("-" * 45)
    
    for samples in tqdm(sample_sizes, desc="Processing"):
        vol, t = estimate_volume_sdf(bunny_path, samples)
        volumes.append(vol)
        times.append(t)
        rel_error = abs(vol - true_volume) / true_volume * 100
        relative_errors.append(rel_error)
        print(f"{samples:^8d} | {vol:^8.8f} | {rel_error:^8.2f} | {t:^7.3f}")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].plot(sample_sizes, relative_errors, 'bo-')
    axes[0].set_xlabel('Number of Samples')
    axes[0].set_ylabel('Relative Error (%)')
    axes[0].set_title('Accuracy vs Samples')
    axes[0].set_xscale('log')
    axes[0].grid(True)
    
    axes[1].plot(sample_sizes, times, 'bo-')
    axes[1].set_xlabel('Number of Samples')
    axes[1].set_ylabel('Computation Time (s)')
    axes[1].set_title('Time vs Samples')
    axes[1].set_xscale('log')
    axes[1].grid(True)
    
    axes[2].plot(times, relative_errors, 'bo-')
    axes[2].set_xlabel('Computation Time (s)')
    axes[2].set_ylabel('Relative Error (%)')
    axes[2].set_title('Accuracy-Efficiency Trade-off')
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig("e:/project/3DV/PAs/PA1/sdf_analysis.png")
    plt.show()

if __name__ == "__main__":
    main()