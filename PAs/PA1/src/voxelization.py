import open3d as o3d
import numpy as np
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
import trimesh

def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024 


def estimate_volume_voxel(mesh_path, resolution):
    initial_memory = get_memory_usage()
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    mesh.compute_vertex_normals()
    is_watertight = mesh.is_watertight()
    if not is_watertight:
        print("Warning: Mesh is not watertight, which may affect volume calculation accuracy")
    start_time = time.time()
    bbox = mesh.get_axis_aligned_bounding_box()
    bbox_size = bbox.get_max_bound() - bbox.get_min_bound()
    voxel_size = max(bbox_size) / resolution
    volume = bbox.volume()
    tri_mesh = trimesh.load(mesh_path)
    tri_mesh.fill_holes()
    tri_mesh.fix_normals()
    voxels = tri_mesh.voxelized(pitch=voxel_size).fill()
    points = voxels.points
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(
        o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(points)),
        voxel_size=voxel_size
    )
    volume = voxels.volume
    time_cost = time.time() - start_time
    memory_used = get_memory_usage() - initial_memory
    
    return volume, time_cost, voxel_grid, memory_used

def compute_true_volume(mesh_path):
    trimesh_mesh = trimesh.load(mesh_path)
    trimesh_mesh.fill_holes()
    trimesh_mesh.fix_normals()
    true_volume = trimesh_mesh.volume
    o3d_mesh = o3d.io.read_triangle_mesh(mesh_path)
    o3d_mesh.compute_vertex_normals()
    return true_volume, o3d_mesh

def main():
    bunny_path = "e:/project/3DV/PAs/PA1/objs_approx/bunny.obj"
    resolutions = [8, 16, 32, 64, 128, 256]
    true_volume, original_mesh = compute_true_volume(bunny_path)
    print(f"True Volume: {true_volume}")
    volumes = []
    times = []
    voxel_grids = []
    relative_errors = []
    memory_usages = []
    print("\nVolume estimation results under different resolutions:")
    print("Resolution | Volume | Error(%) | Time(s) | Memory(MB)")
    print("-" * 45)
    
    for res in tqdm(resolutions, desc="Processing Bunny"):
        vol, t, voxel_grid, mem = estimate_volume_voxel(bunny_path, res)
        volumes.append(vol)
        times.append(t)
        voxel_grids.append(voxel_grid)
        memory_usages.append(mem)
        rel_error = abs(vol - true_volume) / true_volume * 100
        relative_errors.append(rel_error)
        print(f"{res:^10d} | {vol:^10.8f} | {rel_error:^8.2f} | {t:^7.3f} | {mem:^10.2f}")
    
    fig, axes = plt.subplots(1, 4, figsize=(15, 5))
    axes[0].plot(resolutions, relative_errors, 'ro-')
    axes[0].set_xlabel('Resolution')
    axes[0].set_ylabel('Relative Error (%)')
    axes[0].set_title('Accuracy vs Resolution')
    axes[0].grid(True)
    
    axes[1].plot(resolutions, times, 'ro-')
    axes[1].set_xlabel('Resolution')
    axes[1].set_ylabel('Computation Time (s)')
    axes[1].set_title('Time vs Resolution')
    axes[1].grid(True)
    
    axes[2].plot(times, relative_errors, 'ro-')
    axes[2].set_xlabel('Computation Time (s)')
    axes[2].set_ylabel('Relative Error (%)')
    axes[2].set_title('Accuracy-Efficiency Trade-off')
    axes[2].grid(True)
    
    axes[3].plot(resolutions,memory_usages , 'ro-')
    axes[3].set_xlabel('Resolution')
    axes[3].set_ylabel('Memory Usage (MB)')
    axes[3].set_title('Memory Usage vs Resolution')
    axes[3].grid(True)
    
    
    
    plt.tight_layout()
    plt.savefig("e:/project/3DV/PAs/PA1/volume_analysis.png")
    plt.show()
    best_res_idx = np.argmin(relative_errors)
    best_res = resolutions[best_res_idx]
    best_voxel_grid = voxel_grids[best_res_idx]
    
    print(f"\nBest resolution: {best_res}, Relative error: {relative_errors[best_res_idx]:.2f}%")
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Voxelization Result")
    original_mesh.paint_uniform_color([0.8, 0.8, 0.8])
    original_mesh.compute_vertex_normals()
    vis.add_geometry(original_mesh)
    vis.add_geometry(best_voxel_grid)
    opt = vis.get_render_option()
    opt.background_color = np.array([0.1, 0.1, 0.1])
    opt.mesh_show_wireframe = True
    opt.mesh_show_back_face = True
    vis.run()
    vis.destroy_window()

if __name__ == "__main__":
    main()