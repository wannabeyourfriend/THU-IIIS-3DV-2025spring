import numpy as np
import torch
import matplotlib.pyplot as plt
import open3d as o3d
from typing import List, Tuple
import trimesh

def o3d_to_trimesh(o3d_mesh):
    vertices = np.asarray(o3d_mesh.vertices)
    triangles = np.asarray(o3d_mesh.triangles)
    mesh_trimesh = trimesh.Trimesh(vertices=vertices, faces=triangles)
    if o3d_mesh.has_vertex_normals():
        mesh_trimesh.vertex_normals = np.asarray(o3d_mesh.vertex_normals)
    if o3d_mesh.has_vertex_colors():
        mesh_trimesh.visual.vertex_colors = (np.asarray(o3d_mesh.vertex_colors) * 255).astype(np.uint8)
    
    return mesh_trimesh


class OptimizationPlacer:
    def __init__(self, mesh, radius: float, max_spheres: int):
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
        
        # pcd = self.mesh.sample_points_uniformly(number_of_points=num_samples)
        # points = np.asarray(pcd.points)
        # normals = np.asarray(pcd.normals)
        # inward_points = points - normals * self.radius * 1.1  #
        
        # return inward_points
    
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
            existing_spheres = torch.tensor(numpy.array(existing_spheres), device=self.device, dtype=torch.float32)
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
    
            
        
    def place_spheres(self) -> Tuple[List[np.ndarray], List[float]]:
        self.sphere_centers = []
        self.volume_ratios = []
        sample_points = self.sample_points_in_mesh(50)
        self.overall_loss_history = []
        
        for sphere_idx in range(self.max_spheres):
            batch_indices = np.random.choice(len(sample_points), 
                                          min(self.batch_size, len(sample_points)), 
                                          replace=False)
            initial_positions = sample_points[batch_indices]
            optimized_positions, volumes = self.optimize_sphere_positions(initial_positions, sphere_idx)
            best_idx = np.argmax(volumes)
            if volumes[best_idx] <= 0:
                break
                
            self.sphere_centers.append(optimized_positions[best_idx])
            current_volume = self.calculate_total_sphere_volume()
            volume_ratio = current_volume / self.mesh_volume
            self.volume_ratios.append(volume_ratio)
            
            print(f"Ball {sphere_idx+1}: Ratio = {volume_ratio:.4f}")
        self.plot_optimization_history()
        self.plot_volume_ratios()
            
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

    def plot_optimization_history(self):
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
            
        plt.show()
    
    def plot_volume_ratios(self):
        if not self.volume_ratios:
            return
            
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(self.volume_ratios) + 1), self.volume_ratios, 'bo-')
        plt.xlabel('N')
        plt.ylabel('Ratio')
        plt.title('N-Ratio')
        plt.grid(True)
        plt.show()

mesh = o3d.io.read_triangle_mesh("bunny.obj")
trimesh_mesh = trimesh.load("bunny.obj")
mesh.compute_vertex_normals()
placer = OptimizationPlacer(mesh, radius=0.03, max_spheres=10)
centers, volume_ratios = placer.place_spheres()

print(f"\nResult:")
print(f"Number: {centers}")
print(f"Ratio: {volume_ratios[-1]:.4f}")