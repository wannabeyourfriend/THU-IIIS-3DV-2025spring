import os
import torch
import numpy as np
import hydra
from omegaconf import DictConfig
import logging
from pathlib import Path
import time
from tqdm import tqdm
import trimesh

from model import MLPNetwork, Conv3DNetwork
from utils import load_tetrahedral_grid, MarchingTetrahedra, sample_points_from_mesh, chamfer_distance, laplace_regularizer_const

logger = logging.getLogger(__name__)

def save_mesh(vertices, faces, filename):
    """
    Save mesh as obj file
    Args:
        vertices: Vertex coordinates [num_vertices, 3]
        faces: Face indices [num_faces, 3]
        filename: Output file path
    """
    mesh = trimesh.Trimesh(vertices=vertices.detach().cpu().numpy(),
                          faces=faces.detach().cpu().numpy())
    mesh.export(filename)

def save_points(points, filename):
    """
    Save point cloud as ply file
    Args:
        points: Point coordinates [num_points, 3]
        filename: Output file path
    """
    pc = trimesh.PointCloud(points.detach().cpu().numpy())
    pc.export(filename)

def load_point_cloud(file_path):
    """
    Load point cloud data
    Args:
        file_path: Path to point cloud file
    Returns:
        points: Point coordinates [num_points, 3]
    """
    mesh = trimesh.load(file_path)
    if hasattr(mesh, 'vertices'):
        points = mesh.vertices
    else:
        points = np.asarray(mesh.vertices)
    
    return torch.tensor(points, dtype=torch.float32)

def train_epoch(model, optimizer, vertices, tets, target_points, mt_algorithm, device, cfg):
    """
    Train for one epoch
    Args:
        model: Network model
        optimizer: Optimizer
        vertices: Tetrahedral grid vertices [num_vertices, 3]
        tets: Tetrahedron indices [num_tets, 4]
        target_points: Target point cloud [num_points, 3]
        mt_algorithm: Marching Tetrahedra algorithm instance
        device: Computation device
        cfg: Configuration parameters
    Returns:
        loss: Training loss value
        mesh_vertices: Generated mesh vertices
        mesh_faces: Generated mesh faces
        sampled_points: Points sampled from mesh
    """
    model.train()
    optimizer.zero_grad()
    
    deformation, sdf = model(vertices)
    deformed_vertices = vertices + deformation
    
    mesh_vertices, mesh_faces = mt_algorithm(deformed_vertices.unsqueeze(0), 
                                           tets.unsqueeze(0), 
                                           sdf.unsqueeze(0).unsqueeze(-1))
    
    sampled_points = sample_points_from_mesh(mesh_vertices[0], mesh_faces[0], cfg.training.num_samples)
    
    cd_loss = chamfer_distance(sampled_points, target_points.to(device))
    lap_loss = laplace_regularizer_const(mesh_vertices[0], mesh_faces[0])
    loss = cd_loss + cfg.training.lambda_reg * lap_loss
    
    loss.backward()
    optimizer.step()
    
    return loss.item(), mesh_vertices[0], mesh_faces[0], sampled_points

def validate(model, vertices, tets, target_points, mt_algorithm, device, cfg):
    """
    Validate model
    Args:
        model: Network model
        vertices: Tetrahedral grid vertices [num_vertices, 3]
        tets: Tetrahedron indices [num_tets, 4]
        target_points: Target point cloud [num_points, 3]
        mt_algorithm: Marching Tetrahedra algorithm instance
        device: Computation device
        cfg: Configuration parameters
    Returns:
        loss: Validation loss value
        mesh_vertices: Generated mesh vertices
        mesh_faces: Generated mesh faces
        sampled_points: Points sampled from mesh
    """
    model.eval()
    with torch.no_grad():
        deformation, sdf = model(vertices)
        deformed_vertices = vertices + deformation
        
        mesh_vertices, mesh_faces = mt_algorithm(deformed_vertices.unsqueeze(0), 
                                               tets.unsqueeze(0), 
                                               sdf.unsqueeze(0).unsqueeze(-1))
        
        sampled_points = sample_points_from_mesh(mesh_vertices[0], mesh_faces[0], cfg.training.num_samples)
        
        cd_loss = chamfer_distance(sampled_points, target_points.to(device))
        lap_loss = laplace_regularizer_const(mesh_vertices[0], mesh_faces[0])
        loss = cd_loss + cfg.training.lambda_reg * lap_loss
    
    return loss.item(), mesh_vertices[0], mesh_faces[0], sampled_points

@hydra.main(config_path="../configs", config_name="config", version_base="1.2")
def main(cfg: DictConfig):
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    output_dir = Path(cfg.output_dir) / f"{cfg.model.type}_{cfg.data.grid_res}_{cfg.data.point_cloud}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    from omegaconf import OmegaConf
    config_path = output_dir / "config.yaml"
    OmegaConf.save(config=cfg, f=config_path)
    logger.info(f"Experiment configuration saved to {config_path}")
    
    tet_path = os.path.join(cfg.data.tet_dir, f"{cfg.data.grid_res}_compress.npz")
    logger.info(f"Loading tetrahedral grid from {tet_path}")
    vertices_np, tets_np = load_tetrahedral_grid(tet_path)
    
    logger.info(f"Loaded tetrahedral grid with {len(vertices_np)} vertices and {len(tets_np)} tetrahedra")
    
    vertices = torch.tensor(vertices_np, dtype=torch.float32, device=device)
    tets = torch.tensor(tets_np, dtype=torch.long, device=device)
    
    pc_path = os.path.join(cfg.data.pc_dir, f"{cfg.data.point_cloud}_pts.ply")
    logger.info(f"Loading point cloud from {pc_path}")
    target_points = load_point_cloud(pc_path).to(device)
    logger.info(f"Loaded point cloud with {len(target_points)} points")
    
    mt_algorithm = MarchingTetrahedra().to(device)
    
    if cfg.model.type == "mlp":
        model = MLPNetwork(
            pos_enc_freqs=cfg.model.num_frequencies,
            hidden_dim=cfg.model.hidden_dim,
            num_layers=cfg.model.num_layers
        ).to(device)
    elif cfg.model.type == "conv":
        model = Conv3DNetwork(
            grid_size=cfg.data.grid_res,
            pos_enc_freqs=cfg.model.num_frequencies,
            hidden_dim=cfg.model.hidden_dim,
            num_conv_layers=cfg.model.num_layers
        ).to(device)
    else:
        raise ValueError(f"Unknown model type: {cfg.model.type}")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.lr)
    
    best_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(cfg.training.num_epochs):
        start_time = time.time()
        
        train_loss, mesh_vertices, mesh_faces, sampled_points = train_epoch(
            model, optimizer, vertices, tets, target_points, mt_algorithm, device, cfg
        )
        
        do_validation = (epoch + 1) % cfg.training.val_interval == 0 or epoch == 0
        
        if do_validation:
            val_loss, val_mesh_vertices, val_mesh_faces, val_sampled_points = validate(
                model, vertices, tets, target_points, mt_algorithm, device, cfg
            )
            
            epoch_time = time.time() - start_time
            
            logger.info(f"Epoch {epoch+1}/{cfg.training.num_epochs} - "
                       f"Train Loss: {train_loss:.6f}, "
                       f"Val Loss: {val_loss:.6f}, "
                       f"Time: {epoch_time:.2f}s")
            
            if val_loss < best_loss:
                best_loss = val_loss
                patience_counter = 0
                
                torch.save(model.state_dict(), output_dir / "best_model.pth")
                
                save_mesh(val_mesh_vertices, val_mesh_faces, output_dir / "best_mesh.obj")
                save_points(val_sampled_points, output_dir / "best_sampled_points.ply")
                save_points(target_points.cpu(), output_dir / "target_points.ply")
                
                logger.info(f"New best model saved with validation loss: {best_loss:.6f}")
            else:
                patience_counter += 1
                logger.info(f"No improvement for {patience_counter} validations")
        else:
            epoch_time = time.time() - start_time
            logger.info(f"Epoch {epoch+1}/{cfg.training.num_epochs} - "
                       f"Train Loss: {train_loss:.6f}, "
                       f"Time: {epoch_time:.2f}s")
        
        if (epoch + 1) % cfg.training.save_interval == 0:
            torch.save(model.state_dict(), output_dir / f"model_epoch_{epoch+1}.pth")
            save_mesh(mesh_vertices, mesh_faces, output_dir / f"mesh_epoch_{epoch+1}.obj")
            save_points(sampled_points, output_dir / f"sampled_points_epoch_{epoch+1}.ply")
        
        if cfg.training.early_stopping and patience_counter >= cfg.training.patience:
            logger.info(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    logger.info(f"Training completed. Best validation loss: {best_loss:.6f}")
    logger.info(f"Results saved to {output_dir}")

if __name__ == "__main__":
    main()