import os
import torch
import numpy as np
import argparse
import yaml
import open3d as o3d
from tqdm import tqdm

from model import DMTetModel
from utils import load_tetrahedral_grid, MarchingTetrahedra, sample_points_from_mesh, chamfer_distance
from train import save_mesh, save_point_cloud


def visualize_results(mesh_path, sampled_points_path, target_points_path):
    """
    Visualize the results of the reconstruction
    Args:
        mesh_path
        sampled_points_path
        target_points_path
    """
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color([0.7, 0.7, 0.7])
    
    sampled_pcd = o3d.io.read_point_cloud(sampled_points_path)
    sampled_pcd.paint_uniform_color([1, 0, 0])  
    
    target_pcd = o3d.io.read_point_cloud(target_points_path)
    target_pcd.paint_uniform_color([0, 0, 1])  
    
    o3d.visualization.draw_geometries([mesh, sampled_pcd, target_pcd],
                                     window_name="DMTet reconstruction results",
                                     width=800, height=600)


def test_single_object(config, obj_name):
    """
    Args:
        config
        obj_name
    """
    device = torch.device(config['device'])
    
    tet_file = os.path.join(config['tet_dir'], f"{config['grid_res']}_compress.npz")
    vertices, tets = load_tetrahedral_grid(tet_file)
    vertices = torch.from_numpy(vertices).float().to(device)
    tets = torch.from_numpy(tets).long().to(device)
    
    point_file = os.path.join(config['point_cloud_dir'], f"{obj_name}_pts.ply")
    pcd = o3d.io.read_point_cloud(point_file)
    points = np.asarray(pcd.points).astype(np.float32)
    
    center = np.mean(points, axis=0)
    points = points - center
    scale = np.max(np.abs(points))
    points = points / scale
    
    target_points = torch.from_numpy(points).float().to(device).unsqueeze(0)
    
    if config['network_type'] == 'mlp':
        model = DMTetModel(
            network_type='mlp',
            pos_enc_freqs=config['pos_enc_freqs'],
            hidden_dim=config['hidden_dim'],
            num_layers=config['num_layers']
        ).to(device)
    else:  
        model = DMTetModel(
            network_type='conv3d',
            grid_size=config['grid_res'],
            pos_enc_freqs=config['pos_enc_freqs'],
            hidden_dim=config['hidden_dim'],
            num_conv_layers=config['num_layers']
        ).to(device)
    
    model_dir = os.path.join(config['output_dir'], f"{config['network_type']}_{config['grid_res']}")
    model_path = os.path.join(model_dir, f"model_epoch_{config['test_epoch']}.pth")
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    mt = MarchingTetrahedra().to(device)
    
    output_dir = os.path.join(config['output_dir'], f"test_{config['network_type']}_{config['grid_res']}", obj_name)
    os.makedirs(output_dir, exist_ok=True)
    
    with torch.no_grad():
        batch_vertices = vertices.unsqueeze(0)
        batch_tets = tets.unsqueeze(0)
        
        sdf, deform, deformed_vertices = model(batch_vertices)
        
        mesh_vertices, mesh_faces = mt(deformed_vertices, batch_tets, sdf)
        
        sampled_points = sample_points_from_mesh(mesh_vertices, mesh_faces, config['num_samples'])
        
        cd_loss = chamfer_distance(sampled_points, target_points)
        
        mesh_path = os.path.join(output_dir, "mesh.obj")
        save_mesh(mesh_vertices[0], mesh_faces[0], mesh_path)
        
        sampled_points_path = os.path.join(output_dir, "sampled_points.ply")
        save_point_cloud(sampled_points[0], sampled_points_path)
        
        target_points_path = os.path.join(output_dir, "target_points.ply")
        save_point_cloud(target_points[0], target_points_path)
        
        print(f"Object: {obj_name}, CD Loss: {cd_loss.item():.6f}")
        print(f"Results saved to: {output_dir}")
        
        visualize_results(mesh_path, sampled_points_path, target_points_path)


def test_multiple_grids(config, obj_name):
    """
    Different grid resolutions to test
    Args:
        config
        obj_name
    """
    device = torch.device(config['device'])
    point_file = os.path.join(config['point_cloud_dir'], f"{obj_name}_pts.ply")
    pcd = o3d.io.read_point_cloud(point_file)
    points = np.asarray(pcd.points).astype(np.float32)
    center = np.mean(points, axis=0)
    points = points - center
    scale = np.max(np.abs(points))
    points = points / scale
    target_points = torch.from_numpy(points).float().to(device).unsqueeze(0)
    mt = MarchingTetrahedra().to(device)
    
    grid_resolutions = [64, 70, 80, 90, 100, 128]
    results = {}
    
    for grid_res in grid_resolutions:
        tet_file = os.path.join(config['tet_dir'], f"{grid_res}_compress.npz")
        vertices, tets = load_tetrahedral_grid(tet_file)
        vertices = torch.from_numpy(vertices).float().to(device)
        tets = torch.from_numpy(tets).long().to(device)
        
        if config['network_type'] == 'mlp':
            model = DMTetModel(
                network_type='mlp',
                pos_enc_freqs=config['pos_enc_freqs'],
                hidden_dim=config['hidden_dim'],
                num_layers=config['num_layers']
            ).to(device)
        else:  # conv3d
            model = DMTetModel(
                network_type='conv3d',
                grid_size=grid_res,
                pos_enc_freqs=config['pos_enc_freqs'],
                hidden_dim=config['hidden_dim'],
                num_conv_layers=config['num_layers']
            ).to(device)
        
        model_dir = os.path.join(config['output_dir'], f"{config['network_type']}_{grid_res}")
        model_path = os.path.join(model_dir, f"model_epoch_{config['test_epoch']}.pth")
        
        if not os.path.exists(model_path):
            print(f"Model not exists: {model_path}, skip")
            continue
        
        model.load_state_dict(torch.load(model_path))
        model.eval()
        
        output_dir = os.path.join(config['output_dir'], f"compare_{config['network_type']}", obj_name, f"grid_{grid_res}")
        os.makedirs(output_dir, exist_ok=True)
        
        with torch.no_grad():
            batch_vertices = vertices.unsqueeze(0)
            batch_tets = tets.unsqueeze(0)
            
            sdf, deform, deformed_vertices = model(batch_vertices)
            
            mesh_vertices, mesh_faces = mt(deformed_vertices, batch_tets, sdf)
            
            sampled_points = sample_points_from_mesh(mesh_vertices, mesh_faces, config['num_samples'])
            
            cd_loss = chamfer_distance(sampled_points, target_points)
            
            mesh_path = os.path.join(output_dir, "mesh.obj")
            save_mesh(mesh_vertices[0], mesh_faces[0], mesh_path)
            
            sampled_points_path = os.path.join(output_dir, "sampled_points.ply")
            save_point_cloud(sampled_points[0], sampled_points_path)
            
            target_points_path = os.path.join(output_dir, "target_points.ply")
            save_point_cloud(target_points[0], target_points_path)
            
            print(f"Grid Resolution: {grid_res}, CD Loss: {cd_loss.item():.6f}")
            
            results[grid_res] = {
                'cd_loss': cd_loss.item(),
                'mesh_path': mesh_path,
                'sampled_points_path': sampled_points_path,
                'target_points_path': target_points_path
            }
    
    print("\nDifferent resolution results:")
    print("-" * 50)
    print(f"{'Grid Resolution':<20} {'CD Loss':<15}")
    print("-" * 50)
    for grid_res in sorted(results.keys()):
        print(f"{grid_res:<20} {results[grid_res]['cd_loss']:<15.6f}")
    print("-" * 50)


def test_lambda_reg_effect(config, obj_name):
    """
    lambda_reg
    Args:
        config: 
        obj_name: 
    """
    device = torch.device(config['device'])
    
    tet_file = os.path.join(config['tet_dir'], f"{config['grid_res']}_compress.npz")
    vertices, tets = load_tetrahedral_grid(tet_file)
    vertices = torch.from_numpy(vertices).float().to(device)
    tets = torch.from_numpy(tets).long().to(device)
    
    point_file = os.path.join(config['point_cloud_dir'], f"{obj_name}_pts.ply")
    pcd = o3d.io.read_point_cloud(point_file)
    points = np.asarray(pcd.points).astype(np.float32)
    
    center = np.mean(points, axis=0)
    points = points - center
    scale = np.max(np.abs(points))
    points = points / scale
    
    target_points = torch.from_numpy(points).float().to(device).unsqueeze(0)
    
    mt = MarchingTetrahedra().to(device)
    
    lambda_regs = [0.0, 0.01, 0.1, 0.5, 1.0]
    results = {}
    
    for lambda_reg in lambda_regs:
        config_copy = config.copy()
        config_copy['lambda_reg'] = lambda_reg
        
        if config['network_type'] == 'mlp':
            model = DMTetModel(
                network_type='mlp',
                pos_enc_freqs=config['pos_enc_freqs'],
                hidden_dim=config['hidden_dim'],
                num_layers=config['num_layers']
            ).to(device)
        else:  # conv3d
            model = DMTetModel(
                network_type='conv3d',
                grid_size=config['grid_res'],
                pos_enc_freqs=config['pos_enc_freqs'],
                hidden_dim=config['hidden_dim'],
                num_conv_layers=config['num_layers']
            ).to(device)
        
        model_dir = os.path.join(config['output_dir'], f"{config['network_type']}_{config['grid_res']}_lambda_{lambda_reg}")
        model_path = os.path.join(model_dir, f"model_epoch_{config['test_epoch']}.pth")
        
        if not os.path.exists(model_path):
            print(f"Model not exist: {model_path}, skip")
            continue
        
        model.load_state_dict(torch.load(model_path))
        model.eval()
        
        output_dir = os.path.join(config['output_dir'], f"lambda_reg_effect", obj_name, f"lambda_{lambda_reg}")
        os.makedirs(output_dir, exist_ok=True)
        
        with torch.no_grad():
            batch_vertices = vertices.unsqueeze(0)
            batch_tets = tets.unsqueeze(0)
            
            sdf, deform, deformed_vertices = model(batch_vertices)
            
            mesh_vertices, mesh_faces = mt(deformed_vertices, batch_tets, sdf)
            
            sampled_points = sample_points_from_mesh(mesh_vertices, mesh_faces, config['num_samples'])
            
            cd_loss = chamfer_distance(sampled_points, target_points)
            
            mesh_path = os.path.join(output_dir, "mesh.obj")
            save_mesh(mesh_vertices[0], mesh_faces[0], mesh_path)
            
            sampled_points_path = os.path.join(output_dir, "sampled_points.ply")
            save_point_cloud(sampled_points[0], sampled_points_path)
            
            target_points_path = os.path.join(output_dir, "target_points.ply")
            save_point_cloud(target_points[0], target_points_path)
            
            print(f"Lambda Reg: {lambda_reg}, CD Loss: {cd_loss.item():.6f}")
            
            results[lambda_reg] = {
                'cd_loss': cd_loss.item(),
                'mesh_path': mesh_path,
                'sampled_points_path': sampled_points_path,
                'target_points_path': target_points_path
            }
    
    print("\nCompare different lambda_reg:")
    print("-" * 50)
    print(f"{'Lambda Reg':<20} {'CD Loss':<15}")
    print("-" * 50)
    for lambda_reg in sorted(results.keys()):
        print(f"{lambda_reg:<20} {results[lambda_reg]['cd_loss']:<15.6f}")
    print("-" * 50)


def main():
    parser = argparse.ArgumentParser(description="DMTet Test Script")
    parser.add_argument('--config', type=str, default='../configs/config.yaml', help='config file path')
    parser.add_argument('--obj_name', type=str, default='bunny', help='object name')
    parser.add_argument('--mode', type=str, default='single', 
                        choices=['single', 'multi_grid', 'lambda_effect'], 
                        help='Test mode: single, multi_grid, lambda_effect')
    args = parser.parse_args()
    

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    if args.mode == 'single':
        test_single_object(config, args.obj_name)
    elif args.mode == 'multi_grid':
        test_multiple_grids(config, args.obj_name)
    elif args.mode == 'lambda_effect':
        test_lambda_reg_effect(config, args.obj_name)


if __name__ == '__main__':
    main()