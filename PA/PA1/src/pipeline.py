import numpy as np
import torch
import matplotlib.pyplot as plt
import open3d as o3d
from typing import List, Tuple, Dict, Any
import trimesh
import os
import json
import time
from datetime import datetime
import pickle
from tqdm import tqdm

def create_experiment_dirs(base_dir="experiments"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(base_dir, f"experiment_{timestamp}")
    
    os.makedirs(exp_dir, exist_ok=True)
    
    subdirs = ["models", "visualizations", "checkpoints", "logs"]
    paths = {}
    for subdir in subdirs:
        path = os.path.join(exp_dir, subdir)
        os.makedirs(path, exist_ok=True)
        paths[subdir] = path
    
    return exp_dir, paths

def o3d_to_trimesh(o3d_mesh):
    vertices = np.asarray(o3d_mesh.vertices)
    triangles = np.asarray(o3d_mesh.triangles)
    mesh_trimesh = trimesh.Trimesh(vertices=vertices, faces=triangles)
    if o3d_mesh.has_vertex_normals():
        mesh_trimesh.vertex_normals = np.asarray(o3d_mesh.vertex_normals)
    if o3d_mesh.has_vertex_colors():
        mesh_trimesh.visual.vertex_colors = (np.asarray(o3d_mesh.vertex_colors) * 255).astype(np.uint8)
    
    return mesh_trimesh

def visualize_mesh_and_spheres(mesh, sphere_centers, radius, save_path=None, mesh_color=[0.8, 0.8, 0.8, 0.5]):
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Mesh and Spheres Visualization", width=800, height=600)
    
    mesh_copy = o3d.geometry.TriangleMesh(mesh)
    mesh_copy.compute_vertex_normals()
    mesh_copy.paint_uniform_color(mesh_color[:3])
    vis.add_geometry(mesh_copy)
    
    colors = plt.cm.rainbow(np.linspace(0, 1, len(sphere_centers)))[:, :3]
    for i, center in enumerate(sphere_centers):
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
        sphere.translate(center)
        sphere.compute_vertex_normals()
        sphere.paint_uniform_color(colors[i % len(colors)])
        vis.add_geometry(sphere)
    
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=max(mesh.get_max_bound() - mesh.get_min_bound()) * 0.2
    )
    vis.add_geometry(coordinate_frame)
    
    opt = vis.get_render_option()
    opt.background_color = np.asarray([1, 1, 1])
    opt.mesh_show_back_face = True
    opt.point_size = 5.0
    
    ctr = vis.get_view_control()
    ctr.set_zoom(0.8)
    
    if save_path:
        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image(save_path)
    
    vis.run()
    vis.destroy_window()


class OptimizationPlacer:
    def __init__(self, mesh, radius: float, max_spheres: int, exp_dir: str = None, checkpoint_interval: int = 5):
        self.mesh = mesh
        self.radius = radius
        self.max_spheres = max_spheres
        self.sphere_centers = []
        self.learning_rate = 0.01
        self.max_iterations = 100
        self.batch_size = 64
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.overall_loss_history = []
        self.mesh_volume = self.calculate_mesh_volume()
        self.volume_ratios = []
        
        self.exp_dir = exp_dir
        self.checkpoint_interval = checkpoint_interval
        self.experiment_params = {
            "radius": radius,
            "max_spheres": max_spheres,
            "learning_rate": self.learning_rate,
            "max_iterations": self.max_iterations,
            "batch_size": self.batch_size,
            "device": str(self.device),
            "mesh_volume": self.mesh_volume
        }
        
        if self.exp_dir:
            self.save_experiment_params()

    def save_experiment_params(self):
        params_path = os.path.join(self.exp_dir, "logs", "experiment_params.json")
        with open(params_path, 'w') as f:
            json.dump(self.experiment_params, f, indent=4)
        print(f"Experiment parameters saved to {params_path}")

    def calculate_mesh_volume(self) -> float:
        vertices = np.asarray(self.mesh.vertices)
        triangles = np.asarray(self.mesh.triangles)
        mesh_trimesh = trimesh.Trimesh(vertices=vertices, faces=triangles)
        return mesh_trimesh.volume
        
    def sample_points_in_mesh(self, num_samples: int) -> np.ndarray:
        def sample_points_in_mesh_1(mesh, n_points, r):
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            bounds = mesh.bounds
            valid_points = torch.empty((0, 3), device=device)
            while len(valid_points) < n_points:
                points_needed = (n_points - len(valid_points)) * 2
                points = torch.rand((points_needed * 2, 3), device=device)
                points = points * (torch.tensor(bounds[1] - bounds[0], device=device)) + torch.tensor(bounds[0], device=device)
                points_cpu = points.cpu().numpy()
                inside = mesh.contains(points_cpu)
                inside_mask = torch.tensor(inside, device=device)
                inside_points = points[inside_mask]
                if len(inside_points) > 0:
                    distances = torch.tensor(
                        mesh.nearest.signed_distance(inside_points.cpu().numpy()),
                        device=device
                    )
                    
                    far_enough = distances >= r
                    valid_new_points = inside_points[far_enough]
                    valid_points = torch.cat((valid_points, valid_new_points)) if len(valid_points) > 0 else valid_new_points
            if len(valid_points) > n_points:
                indices = torch.randperm(len(valid_points), device=device)[:n_points]
                valid_points = valid_points[indices]
            return valid_points.cpu().numpy()

        return sample_points_in_mesh_1(o3d_to_trimesh(self.mesh), num_samples, self.radius)
    
    def is_sphere_inside_mesh(self, center: np.ndarray) -> bool:
        bbox = self.mesh.get_axis_aligned_bounding_box()
        min_bound = np.asarray(bbox.min_bound)
        max_bound = np.asarray(bbox.max_bound)
        if np.any(center - self.radius < min_bound) or np.any(center + self.radius > max_bound):
            return False
        sphere = trimesh.primitives.Sphere(radius=self.radius, center=center)
        vertices = np.asarray(self.mesh.vertices)
        triangles = np.asarray(self.mesh.triangles)
        mesh_trimesh = trimesh.Trimesh(vertices=vertices, faces=triangles)
        if mesh_trimesh.contains([center]):
            return True
        
        surface_points = []
        for axis in range(3):
            for direction in [-1, 1]:
                offset = np.zeros(3)
                offset[axis] = direction * self.radius
                surface_points.append(center + offset)
        return all(mesh_trimesh.contains(surface_points))
    
    def calculate_batch_volumes(self, positions: np.ndarray, existing_spheres: List[np.ndarray]) -> np.ndarray:
        positions = torch.tensor(positions, device=self.device, dtype=torch.float32)
        
        if len(existing_spheres) > 0:
            existing_spheres = torch.tensor(existing_spheres, device=self.device, dtype=torch.float32)
            distances = torch.sqrt(((positions.unsqueeze(1) - existing_spheres.unsqueeze(0)) ** 2).sum(dim=2))
            overlaps = torch.clamp(2 * self.radius - distances, min=0)
            overlap_volumes = (torch.pi / 6) * overlaps ** 3
            total_overlaps = overlap_volumes.sum(dim=1)
        else:
            total_overlaps = torch.zeros(len(positions), device=self.device)
        
        sphere_volume = (4/3) * torch.pi * self.radius ** 3
        total_volumes = sphere_volume * (len(existing_spheres) + 1) - total_overlaps
        
        return total_volumes.cpu().numpy()

    def calculate_total_sphere_volume(self) -> float:
        if not self.sphere_centers:
            return 0.0
        if len(self.sphere_centers) == 1:
            return (4/3) * np.pi * self.radius**3
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        centers = np.array(self.sphere_centers)
        radii = np.ones(len(centers)) * self.radius
        centers_torch = torch.from_numpy(centers).float().to(device)
        radii_torch = torch.from_numpy(radii).float().to(device)
        
        bbox_min = torch.from_numpy(centers.min(axis=0) - self.radius).float().to(device)
        bbox_max = torch.from_numpy(centers.max(axis=0) + self.radius).float().to(device)
        
        num_samples = 10000
        
        points = torch.empty(num_samples, 3, device=device).uniform_(0, 1)
        points = points * (bbox_max - bbox_min) + bbox_min
        distances = torch.norm(
            points.unsqueeze(1) - centers_torch.unsqueeze(0),
            dim=2
        )
        inside_any = (distances <= radii_torch.unsqueeze(0)).any(dim=1)
        
        bbox_volume = torch.prod(bbox_max - bbox_min).item()
        ratio = torch.sum(inside_any.float()).item() / num_samples
        
        return bbox_volume * ratio
    
    def save_checkpoint(self, sphere_idx: int):
        if not self.exp_dir:
            return
            
        checkpoint = {
            "sphere_centers": self.sphere_centers,
            "volume_ratios": self.volume_ratios,
            "overall_loss_history": self.overall_loss_history,
            "current_sphere_idx": sphere_idx,
            "experiment_params": self.experiment_params
        }
        
        checkpoint_path = os.path.join(self.exp_dir, "checkpoints", f"checkpoint_sphere_{sphere_idx}.pkl")
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(checkpoint, f)
        print(f"Checkpoint saved to {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        with open(checkpoint_path, 'rb') as f:
            checkpoint = pickle.load(f)
        
        self.sphere_centers = checkpoint["sphere_centers"]
        self.volume_ratios = checkpoint["volume_ratios"]
        self.overall_loss_history = checkpoint["overall_loss_history"]
        
        return checkpoint["current_sphere_idx"] + 1
        
    def place_spheres(self, start_sphere_idx: int = 0, sample_points: np.ndarray = None) -> Tuple[List[np.ndarray], List[float]]:
        if start_sphere_idx == 0:
            self.sphere_centers = []
            self.volume_ratios = []
            self.overall_loss_history = []
        
        if sample_points is None:
            sample_points = self.sample_points_in_mesh(100)
            
        for sphere_idx in tqdm(range(start_sphere_idx, self.max_spheres), desc="Placing spheres"):
            batch_indices = np.random.choice(len(sample_points), 
                                          min(self.batch_size, len(sample_points)), 
                                          replace=False)
            initial_positions = sample_points[batch_indices]
            optimized_positions, volumes = self.optimize_sphere_positions(initial_positions, sphere_idx)
            best_idx = np.argmax(volumes)
            if volumes[best_idx] <= 0:
                print(f"Cannot place more spheres, placed {sphere_idx} spheres")
                break
                
            self.sphere_centers.append(optimized_positions[best_idx])
            current_volume = self.calculate_total_sphere_volume()
            volume_ratio = current_volume / self.mesh_volume
            self.volume_ratios.append(volume_ratio)
            
            print(f"Sphere {sphere_idx+1}: Volume ratio = {volume_ratio:.4f}")
            
            if self.exp_dir and (sphere_idx + 1) % 5 == 0:
                vis_path = os.path.join(self.exp_dir, "visualizations", f"spheres_{sphere_idx+1}.png")
                visualize_mesh_and_spheres(self.mesh, self.sphere_centers, self.radius, save_path=vis_path)
            
            if self.exp_dir and (sphere_idx + 1) % self.checkpoint_interval == 0:
                self.save_checkpoint(sphere_idx)
                
        if self.exp_dir:
            self.plot_optimization_history(save=True)
            self.plot_volume_ratios(save=True)
            self.save_final_results()
            
        return self.sphere_centers, self.volume_ratios

    def optimize_sphere_positions(self, initial_positions: np.ndarray, sphere_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        positions = torch.tensor(initial_positions, device=self.device, 
                               dtype=torch.float32, requires_grad=True)
        optimizer = torch.optim.Adam([positions], lr=self.learning_rate)
        
        best_positions = positions.detach().clone()
        best_volumes = torch.zeros(len(initial_positions), device=self.device)
        
        for iter_idx in range(self.max_iterations):
            optimizer.zero_grad()
            positions_np = positions.detach().cpu().numpy()
            volumes = self.calculate_batch_volumes(positions_np, self.sphere_centers)
            volumes_tensor = torch.tensor(volumes, device=self.device)
            positions_proxy = positions.sum(dim=1)
            loss = -torch.mean(positions_proxy * volumes_tensor / positions_proxy.detach())
            
            loss.backward()
            self.overall_loss_history.append((sphere_idx, iter_idx, loss.item()))
            optimizer.step()
            with torch.no_grad():
                valid_mask = torch.tensor([self.is_sphere_inside_mesh(p.cpu().numpy()) 
                                         for p in positions], device=self.device)
                if not valid_mask.any():
                    break
                
                better_mask = volumes_tensor > best_volumes
                best_positions[better_mask] = positions[better_mask].clone()
                best_volumes[better_mask] = volumes_tensor[better_mask]
        
        return best_positions.cpu().numpy(), best_volumes.cpu().numpy()

    def plot_optimization_history(self, save=False):
        if not self.overall_loss_history:
            return
            
        sphere_indices = [item[0] for item in self.overall_loss_history]
        iter_indices = [item[1] for item in self.overall_loss_history]
        losses = [item[2] for item in self.overall_loss_history]
        
        global_indices = []
        for s, i in zip(sphere_indices, iter_indices):
            global_indices.append(s * self.max_iterations + i)
            
        plt.figure(figsize=(12, 6))
        plt.plot(global_indices, losses)
        plt.xlabel('Global Iteration')
        plt.ylabel('Loss')
        plt.title('Overall Optimization Process')
        plt.grid(True)
        max_sphere = max(sphere_indices)
        for s in range(max_sphere + 1):
            plt.axvline(x=s * self.max_iterations, color='r', linestyle='--', alpha=0.3)
            plt.text(s * self.max_iterations, min(losses), f'Sphere {s+1}', 
                   rotation=90, verticalalignment='bottom')
        
        if save and self.exp_dir:
            plt.savefig(os.path.join(self.exp_dir, "visualizations", "optimization_history.png"), dpi=300)
            
        plt.show()
    
    def plot_volume_ratios(self, save=False):
        if not self.volume_ratios:
            return
            
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(self.volume_ratios) + 1), self.volume_ratios, 'bo-')
        plt.xlabel('Number of Spheres')
        plt.ylabel('Volume Ratio')
        plt.title('Relationship Between Number of Spheres and Volume Ratio')
        plt.grid(True)
        
        if save and self.exp_dir:
            plt.savefig(os.path.join(self.exp_dir, "visualizations", "volume_ratios.png"), dpi=300)
            
        plt.show()
        
    def save_final_results(self):
        if not self.exp_dir:
            return
            
        centers_path = os.path.join(self.exp_dir, "models", "sphere_centers.npy")
        np.save(centers_path, np.array(self.sphere_centers))
        
        ratios_path = os.path.join(self.exp_dir, "models", "volume_ratios.npy")
        np.save(ratios_path, np.array(self.volume_ratios))
        
        vis_path = os.path.join(self.exp_dir, "visualizations", "final_result.png")
        visualize_mesh_and_spheres(self.mesh, self.sphere_centers, self.radius, save_path=vis_path)
        
        summary = {
            "num_spheres": len(self.sphere_centers),
            "final_volume_ratio": self.volume_ratios[-1] if self.volume_ratios else 0,
            "radius": self.radius,
            "mesh_volume": self.mesh_volume,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        summary_path = os.path.join(self.exp_dir, "logs", "results_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=4)
            
        print(f"Final results saved to {self.exp_dir}")


def run_experiment(mesh_path, radius, max_spheres, checkpoint_path=None):
    exp_dir, paths = create_experiment_dirs()
    
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    mesh.compute_vertex_normals()
    
    placer = OptimizationPlacer(mesh, radius=radius, max_spheres=max_spheres, exp_dir=exp_dir)
    
    start_sphere_idx = 0
    if checkpoint_path:
        start_sphere_idx = placer.load_checkpoint(checkpoint_path)
        print(f"Resumed from checkpoint, starting from sphere {start_sphere_idx}")
    
    centers, volume_ratios = placer.place_spheres(start_sphere_idx=start_sphere_idx)
    
    print(f"\nFinal Results:")
    print(f"Number of spheres placed: {len(centers)}")
    print(f"Final volume ratio: {volume_ratios[-1]:.4f}")
    
    return exp_dir, centers, volume_ratios

def run_parameter_sweep(mesh_path, radii, max_spheres_list):
    results = []
    
    for radius in radii:
        for max_spheres in max_spheres_list:
            print(f"\nStarting experiment: radius={radius}, max_spheres={max_spheres}")
            exp_dir, centers, volume_ratios = run_experiment(mesh_path, radius, max_spheres)
            
            results.append({
                "radius": radius,
                "max_spheres": max_spheres,
                "num_spheres_placed": len(centers),
                "final_volume_ratio": volume_ratios[-1] if volume_ratios else 0,
                "exp_dir": exp_dir
            })
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = f"e:/project/3DV/PAs/PA1/parameter_sweep_results_{timestamp}.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"\nParameter sweep results saved to {results_path}")
    
    plot_parameter_sweep_results(results)
    
    return results

def plot_parameter_sweep_results(results):
    radii = sorted(list(set([r["radius"] for r in results])))
    max_spheres_list = sorted(list(set([r["max_spheres"] for r in results])))
    
    volume_ratios = np.zeros((len(radii), len(max_spheres_list)))
    num_spheres = np.zeros((len(radii), len(max_spheres_list)))
    
    for result in results:
        r_idx = radii.index(result["radius"])
        m_idx = max_spheres_list.index(result["max_spheres"])
        volume_ratios[r_idx, m_idx] = result["final_volume_ratio"]
        num_spheres[r_idx, m_idx] = result["num_spheres_placed"]
    
    plt.figure(figsize=(12, 8))
    plt.subplot(1, 2, 1)
    im = plt.imshow(volume_ratios, cmap='viridis')
    plt.colorbar(im, label='Volume Ratio')
    plt.xticks(range(len(max_spheres_list)), max_spheres_list)
    plt.yticks(range(len(radii)), [f"{r:.3f}" for r in radii])
    plt.xlabel('Maximum Number of Spheres')
    plt.ylabel('Sphere Radius')
    plt.title('Volume Ratio for Different Parameters')
    
    plt.subplot(1, 2, 2)
    im = plt.imshow(num_spheres, cmap='plasma')
    plt.colorbar(im, label='Number of Spheres Placed')
    plt.xticks(range(len(max_spheres_list)), max_spheres_list)
    plt.yticks(range(len(radii)), [f"{r:.3f}" for r in radii])
    plt.xlabel('Maximum Number of Spheres')
    plt.ylabel('Sphere Radius')
    plt.title('Number of Spheres Placed for Different Parameters')
    
    plt.tight_layout()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f"e:/project/3DV/PAs/PA1/parameter_sweep_plot_{timestamp}.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    mesh_path = "bunny.obj"
    
    radii = [0.01, 0.02, 0.03, 0.04]
    max_spheres_list = [20, 50, 100, 200]
    results = run_parameter_sweep(mesh_path, radii, max_spheres_list)