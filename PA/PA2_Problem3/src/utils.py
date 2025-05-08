import torch
import torch.nn.functional as F
import numpy as np
import kaolin

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


class MarchingTetrahedra(torch.nn.Module):
    """
    Differentiable implementation of Marching Tetrahedra algorithm
    Converts SDF values to triangle mesh
    """
    def __init__(self):
        super(MarchingTetrahedra, self).__init__()
        self.triangle_table, self.num_triangles_table, self.base_tet_edges, self.v_id = self._create_mt_variables()
        
    def _create_mt_variables(self):
        """
        Create lookup tables and variables for Marching Tetrahedra
        """
        device = 'cpu'
        triangle_table = torch.tensor(
            [
                [-1, -1, -1, -1, -1, -1],
                [1, 0, 2, -1, -1, -1],
                [4, 0, 3, -1, -1, -1],
                [1, 4, 2, 1, 3, 4],
                [3, 1, 5, -1, -1, -1],
                [2, 3, 0, 2, 5, 3],
                [1, 4, 0, 1, 5, 4],
                [4, 2, 5, -1, -1, -1],
                [4, 5, 2, -1, -1, -1],
                [4, 1, 0, 4, 5, 1],
                [3, 2, 0, 3, 5, 2],
                [1, 3, 5, -1, -1, -1],
                [4, 1, 2, 4, 3, 1],
                [3, 0, 4, -1, -1, -1],
                [2, 0, 1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1]
            ], dtype=torch.long, device=device)
        
        num_triangles_table = torch.tensor(
            [0, 1, 1, 2, 1, 2, 2, 1, 1, 2, 2, 1, 2, 1, 1, 0], 
            dtype=torch.long, device=device
        )
        
        base_tet_edges = torch.tensor(
            [0, 1, 0, 2, 0, 3, 1, 2, 1, 3, 2, 3], 
            dtype=torch.long, device=device
        )
        
        v_id = torch.pow(2, torch.arange(4, dtype=torch.long, device=device))
        
        return triangle_table, num_triangles_table, base_tet_edges, v_id
    
    def _sort_edges(self, edges_ex2):
        """
        Sort edges to ensure vertices are in order
        """
        with torch.no_grad():
            order = (edges_ex2[:, 0] > edges_ex2[:, 1]).long()
            order = order.unsqueeze(dim=1)
            a = torch.gather(input=edges_ex2, index=order, dim=1)
            b = torch.gather(input=edges_ex2, index=1 - order, dim=1)
        return torch.stack([a, b], -1)
    
    def forward(self, vertices, tets, sdf):
        """
        Execute Marching Tetrahedra algorithm
        Args:
            vertices: Vertex coordinates [batch_size, num_vertices, 3]
            tets: Tetrahedron indices [batch_size, num_tets, 4]
            sdf: SDF values per vertex [batch_size, num_vertices, 1]
        Returns:
            mesh_vertices: Mesh vertices [batch_size, num_mesh_vertices, 3]
            mesh_faces: Mesh faces [batch_size, num_mesh_faces, 3]
        """
        batch_size = vertices.shape[0]
        device = vertices.device
        
        triangle_table = self.triangle_table.to(device)
        num_triangles_table = self.num_triangles_table.to(device)
        base_tet_edges = self.base_tet_edges.to(device)
        v_id = self.v_id.to(device)
        
        mesh_vertices = []
        mesh_faces = []
        
        for b in range(batch_size):
            pos_nx3 = vertices[b]
            sdf_n = sdf[b].squeeze(-1)
            tet_fx4 = tets[b]
            
            with torch.no_grad():
                occ_n = sdf_n > 0
                occ_fx4 = occ_n[tet_fx4.reshape(-1)].reshape(-1, 4)
                occ_sum = torch.sum(occ_fx4, -1)
                valid_tets = (occ_sum > 0) & (occ_sum < 4)
                
                if not valid_tets.any():
                    mesh_vertices.append(torch.zeros((0, 3), device=device))
                    mesh_faces.append(torch.zeros((0, 3), dtype=torch.long, device=device))
                    continue
                
                all_edges = tet_fx4[valid_tets][:, base_tet_edges.reshape(-1)].reshape(-1, 2)
                all_edges = self._sort_edges(all_edges)
                unique_edges, idx_map = torch.unique(all_edges, dim=0, return_inverse=True)
                
                unique_edges = unique_edges.long()
                mask_edges = occ_n[unique_edges.reshape(-1)].reshape(-1, 2).sum(-1) == 1
                mapping = torch.ones((unique_edges.shape[0]), dtype=torch.long, device=device) * -1
                mapping[mask_edges] = torch.arange(mask_edges.sum(), dtype=torch.long, device=device)
                idx_map = mapping[idx_map]
                
                interp_v = unique_edges[mask_edges]
            
            edges_to_interp = pos_nx3[interp_v.reshape(-1)].reshape(-1, 2, 3)
            edges_to_interp_sdf = sdf_n[interp_v.reshape(-1)].reshape(-1, 2, 1)
            edges_to_interp_sdf[:, -1] *= -1
            
            denominator = edges_to_interp_sdf.sum(1, keepdim=True)
            edges_to_interp_sdf = torch.flip(edges_to_interp_sdf, [1]) / denominator
            verts = (edges_to_interp * edges_to_interp_sdf).sum(1)
            
            idx_map = idx_map.reshape(-1, 6)
            
            tetindex = (occ_fx4[valid_tets] * v_id.unsqueeze(0)).sum(-1)
            num_triangles = num_triangles_table[tetindex]
            
            faces_list = []
            
            if (num_triangles == 1).any():
                faces_1 = torch.gather(
                    input=idx_map[num_triangles == 1], dim=1,
                    index=triangle_table[tetindex[num_triangles == 1]][:, :3]
                ).reshape(-1, 3)
                faces_list.append(faces_1)
            
            if (num_triangles == 2).any():
                faces_2 = torch.gather(
                    input=idx_map[num_triangles == 2], dim=1,
                    index=triangle_table[tetindex[num_triangles == 2]][:, :6]
                ).reshape(-1, 3)
                faces_list.append(faces_2)
            
            if faces_list:
                faces = torch.cat(faces_list, dim=0)
                mesh_vertices.append(verts)
                mesh_faces.append(faces)
            else:
                mesh_vertices.append(torch.zeros((0, 3), device=device))
                mesh_faces.append(torch.zeros((0, 3), dtype=torch.long, device=device))
        
        return mesh_vertices, mesh_faces

def sample_points_from_mesh(vertices, faces, num_samples=5000):
    """
    Sample points uniformly from mesh surface
    Args:
        vertices: Vertex coordinates [num_vertices, 3]
        faces: Triangle face indices [num_faces, 3]
        num_samples: Number of points to sample
    Returns:
        points: Sampled point coordinates [num_samples, 3]
    """
    device = vertices.device
    
    if faces.shape[0] == 0:
        return torch.zeros((num_samples, 3), device=device)
    
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    
    triangles = torch.stack([v0, v1, v2], dim=1)
    vectors = triangles[:, 1:] - triangles[:, :1]
    cross_products = torch.cross(vectors[:, 0], vectors[:, 1], dim=-1)
    areas = torch.norm(cross_products, dim=-1) * 0.5
    
    if torch.all(areas < 1e-10):
        indices = torch.randint(0, faces.shape[0], (num_samples,), device=device)
        areas = torch.ones_like(areas)
    else:
        prob = areas / areas.sum()
        indices = torch.multinomial(prob, num_samples, replacement=True)
    
    r1 = torch.sqrt(torch.rand(num_samples, device=device))
    r2 = torch.rand(num_samples, device=device)
    
    sampled_v0 = v0[indices]
    sampled_v1 = v1[indices]
    sampled_v2 = v2[indices]
    
    w1 = 1 - r1
    w2 = r1 * (1 - r2)
    w3 = r1 * r2
    
    points = (w1.unsqueeze(-1) * sampled_v0 +
             w2.unsqueeze(-1) * sampled_v1 +
             w3.unsqueeze(-1) * sampled_v2)
    
    return points

def chamfer_distance(x, y, chunk_size=2048):
    """
    Compute Chamfer Distance in chunks to save memory
    """
    if isinstance(x, list):
        return torch.stack([chamfer_distance(x_i, y_i) for x_i, y_i in zip(x, y)])
    
    total_dist = 0.0
    for i in range(0, x.shape[0], chunk_size):
        x_chunk = x[i:i+chunk_size]
        x_expanded = x_chunk.unsqueeze(1)
        y_expanded = y.unsqueeze(0)
        dist_matrix = torch.sum((x_expanded - y_expanded) ** 2, dim=2)
        min_dist = torch.min(dist_matrix, dim=1)[0]
        total_dist += torch.sum(min_dist)
    
    for j in range(0, y.shape[0], chunk_size):
        y_chunk = y[j:j+chunk_size]
        y_expanded = y_chunk.unsqueeze(1)
        x_expanded = x.unsqueeze(0)
        dist_matrix = torch.sum((y_expanded - x_expanded) ** 2, dim=2)
        min_dist = torch.min(dist_matrix, dim=1)[0]
        total_dist += torch.sum(min_dist)
    
    return total_dist / (x.shape[0] + y.shape[0])

def laplace_regularizer_const(mesh_verts, mesh_faces):
    """
    Compute Laplacian regularization term
    Args:
        mesh_verts: Mesh vertices [num_vertices, 3]
        mesh_faces: Mesh faces [num_faces, 3]
    Returns:
        reg: Laplacian regularization value
    """
    if isinstance(mesh_verts, list):
        return torch.stack([laplace_regularizer_const(v, f) for v, f in zip(mesh_verts, mesh_faces)])
    
    term = torch.zeros_like(mesh_verts)
    norm = torch.zeros_like(mesh_verts[..., 0:1])
    if mesh_faces.shape[0] == 0:
        return torch.tensor(0.0, device=mesh_verts.device)
    v0 = mesh_verts[mesh_faces[:, 0], :]
    v1 = mesh_verts[mesh_faces[:, 1], :]
    v2 = mesh_verts[mesh_faces[:, 2], :]
    term.scatter_add_(0, mesh_faces[:, 0:1].repeat(1,3), (v1 - v0) + (v2 - v0))
    term.scatter_add_(0, mesh_faces[:, 1:2].repeat(1,3), (v0 - v1) + (v2 - v1))
    term.scatter_add_(0, mesh_faces[:, 2:3].repeat(1,3), (v0 - v2) + (v1 - v2))
    two = torch.ones_like(v0) * 2.0
    norm.scatter_add_(0, mesh_faces[:, 0:1], two)
    norm.scatter_add_(0, mesh_faces[:, 1:2], two)
    norm.scatter_add_(0, mesh_faces[:, 2:3], two)
    term = term / torch.clamp(norm, min=1.0)
    
    return torch.mean(term**2)

def visualize_point_cloud(points, output_path, title="Point Cloud"):
    """
    Visualize point cloud and save as image
    Args:
        points: Point coordinates [num_points, 3]
        output_path: Path to save the visualization
        title: Title of the plot
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    if torch.is_tensor(points):
        points = points.detach().cpu().numpy()
    
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='b', marker='.', s=1)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    
    ax.view_init(elev=30, azim=45)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()