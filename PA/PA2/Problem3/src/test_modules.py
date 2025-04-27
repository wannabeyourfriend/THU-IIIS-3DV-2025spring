import os
import torch
import numpy as np
import open3d as o3d
from tqdm import tqdm
import hydra
from omegaconf import DictConfig, OmegaConf
import sys

from model import DMTetModel
from utils import load_tetrahedral_grid, MarchingTetrahedra, sample_points_from_mesh, chamfer_distance, laplace_regularizer_const
from data import PointCloudDataset
from train import save_mesh, save_point_cloud

def print_separator(title):
    """打印分隔符和标题"""
    print("\n" + "=" * 50)
    print(f" {title} ".center(50, "="))
    print("=" * 50)

def test_data_loading(config):
    """
    测试点云数据加载
    """
    print_separator("测试点云数据加载")
    
    try:
        # 创建数据集
        dataset = PointCloudDataset(config.point_cloud_dir)
        print(f"数据集大小: {len(dataset)}")
        
        # 获取一个样本
        sample = dataset[0]
        points = sample['points']
        file_name = sample['file_name']
        center = sample['center']
        scale = sample['scale']
        
        print(f"文件名: {file_name}")
        print(f"点云形状: {points.shape}")
        print(f"中心点: {center}")
        print(f"缩放因子: {scale}")
        
        # 验证点云是否已归一化
        min_coords = torch.min(points, dim=0)[0]
        max_coords = torch.max(points, dim=0)[0]
        print(f"点云坐标范围: [{min_coords}, {max_coords}]")
        
        print("✓ 点云数据加载测试通过")
        return True
    except Exception as e:
        print(f"✗ 点云数据加载测试失败: {str(e)}")
        return False

def test_tetrahedral_grid_loading(config):
    """
    测试四面体网格加载
    """
    print_separator("测试四面体网格加载")
    
    try:
        # 加载四面体网格
        tet_file = os.path.join(config.tet_dir, f"{config.grid_res}_compress.npz")
        vertices, tets = load_tetrahedral_grid(tet_file)
        
        print(f"四面体网格文件: {tet_file}")
        print(f"顶点数量: {vertices.shape[0]}")
        print(f"顶点形状: {vertices.shape}")
        print(f"四面体数量: {tets.shape[0]}")
        print(f"四面体形状: {tets.shape}")
        
        # 验证四面体索引是否有效
        max_vertex_idx = np.max(tets)
        if max_vertex_idx >= vertices.shape[0]:
            print(f"✗ 四面体索引超出顶点范围: {max_vertex_idx} >= {vertices.shape[0]}")
            return False
        
        print("✓ 四面体网格加载测试通过")
        return True
    except Exception as e:
        print(f"✗ 四面体网格加载测试失败: {str(e)}")
        return False

def test_model_forward(config):
    """
    测试模型前向传播
    """
    print_separator("测试模型前向传播")
    
    try:
        device = torch.device(config.device)
        
        # 加载四面体网格
        tet_file = os.path.join(config.tet_dir, f"{config.grid_res}_compress.npz")
        vertices, tets = load_tetrahedral_grid(tet_file)
        vertices = torch.from_numpy(vertices).float().to(device)
        tets = torch.from_numpy(tets).long().to(device)
        
        # 创建模型
        if config.network_type == 'mlp':
            model = DMTetModel(
                network_type='mlp',
                pos_enc_freqs=config.pos_enc_freqs,
                hidden_dim=config.hidden_dim,
                num_layers=config.num_layers
            ).to(device)
            print(f"创建MLP网络: pos_enc_freqs={config.pos_enc_freqs}, hidden_dim={config.hidden_dim}, num_layers={config.num_layers}")
        else:  # conv3d
            model = DMTetModel(
                network_type='conv3d',
                grid_size=config.grid_res,
                pos_enc_freqs=config.pos_enc_freqs,
                hidden_dim=config.hidden_dim,
                num_conv_layers=config.num_layers
            ).to(device)
            print(f"创建Conv3D网络: grid_size={config.grid_res}, pos_enc_freqs={config.pos_enc_freqs}, hidden_dim={config.hidden_dim}, num_conv_layers={config.num_layers}")
        
        # 前向传播
        batch_vertices = vertices.unsqueeze(0)  # [1, num_vertices, 3]
        batch_tets = tets.unsqueeze(0)  # [1, num_tets, 4]
        
        print(f"输入顶点形状: {batch_vertices.shape}")
        print(f"输入四面体形状: {batch_tets.shape}")
        
        sdf, deform, deformed_vertices = model(batch_vertices)
        
        print(f"SDF形状: {sdf.shape}")
        print(f"变形向量形状: {deform.shape}")
        print(f"变形后顶点形状: {deformed_vertices.shape}")
        
        # 验证输出形状
        if sdf.shape != (1, vertices.shape[0], 1):
            print(f"✗ SDF形状不正确: {sdf.shape} != {(1, vertices.shape[0], 1)}")
            return False
        
        if deform.shape != (1, vertices.shape[0], 3):
            print(f"✗ 变形向量形状不正确: {deform.shape} != {(1, vertices.shape[0], 3)}")
            return False
        
        if deformed_vertices.shape != (1, vertices.shape[0], 3):
            print(f"✗ 变形后顶点形状不正确: {deformed_vertices.shape} != {(1, vertices.shape[0], 3)}")
            return False
        
        print("✓ 模型前向传播测试通过")
        return True
    except Exception as e:
        print(f"✗ 模型前向传播测试失败: {str(e)}")
        return False

def test_marching_tetrahedra(config):
    """
    测试Marching Tetrahedra算法
    """
    print_separator("测试Marching Tetrahedra算法")
    
    try:
        device = torch.device(config.device)
        
        # 加载四面体网格
        tet_file = os.path.join(config.tet_dir, f"{config.grid_res}_compress.npz")
        vertices, tets = load_tetrahedral_grid(tet_file)
        vertices = torch.from_numpy(vertices).float().to(device)
        tets = torch.from_numpy(tets).long().to(device)
        
        # 创建随机SDF值
        sdf = torch.randn(1, vertices.shape[0], 1, device=device)
        
        # 创建Marching Tetrahedra算法
        mt = MarchingTetrahedra().to(device)
        
        # 执行算法
        batch_vertices = vertices.unsqueeze(0)
        batch_tets = tets.unsqueeze(0)
        
        print(f"输入顶点形状: {batch_vertices.shape}")
        print(f"输入四面体形状: {batch_tets.shape}")
        print(f"输入SDF形状: {sdf.shape}")
        
        mesh_vertices, mesh_faces = mt(batch_vertices, batch_tets, sdf)
        
        # 验证输出
        if isinstance(mesh_vertices, list):
            print(f"网格顶点数量: {len(mesh_vertices[0])}")
            print(f"网格面片数量: {len(mesh_faces[0])}")
        else:
            print(f"网格顶点形状: {mesh_vertices.shape}")
            print(f"网格面片形状: {mesh_faces.shape}")
        
        print("✓ Marching Tetrahedra算法测试通过")
        return True
    except Exception as e:
        print(f"✗ Marching Tetrahedra算法测试失败: {str(e)}")
        return False

def test_point_sampling(config):
    """
    测试点云采样
    """
    print_separator("测试点云采样")
    
    try:
        device = torch.device(config.device)
        
        # 创建一个简单的立方体网格
        vertices = torch.tensor([
            [-1, -1, -1],
            [1, -1, -1],
            [1, 1, -1],
            [-1, 1, -1],
            [-1, -1, 1],
            [1, -1, 1],
            [1, 1, 1],
            [-1, 1, 1]
        ], dtype=torch.float32, device=device)
        
        faces = torch.tensor([
            [0, 1, 2], [0, 2, 3],  # 底面
            [4, 5, 6], [4, 6, 7],  # 顶面
            [0, 1, 5], [0, 5, 4],  # 前面
            [2, 3, 7], [2, 7, 6],  # 后面
            [0, 3, 7], [0, 7, 4],  # 左面
            [1, 2, 6], [1, 6, 5]   # 右面
        ], dtype=torch.long, device=device)
        
        # 采样点
        num_samples = config.num_samples
        sampled_points = sample_points_from_mesh(vertices, faces, num_samples)
        
        print(f"采样点数量: {num_samples}")
        print(f"采样点形状: {sampled_points.shape}")
        
        # 验证采样点是否在立方体内部或表面
        min_coords = torch.min(sampled_points, dim=0)[0]
        max_coords = torch.max(sampled_points, dim=0)[0]
        print(f"采样点坐标范围: [{min_coords}, {max_coords}]")
        
        # 验证采样点数量
        if sampled_points.shape[0] != num_samples:
            print(f"✗ 采样点数量不正确: {sampled_points.shape[0]} != {num_samples}")
            return False
        
        print("✓ 点云采样测试通过")
        return True
    except Exception as e:
        print(f"✗ 点云采样测试失败: {str(e)}")
        return False

def test_loss_functions(config):
    """
    测试损失函数
    """
    print_separator("测试损失函数")
    
    try:
        device = torch.device(config.device)
        
        # 创建随机点云
        num_points = 1000
        source_points = torch.randn(1, num_points, 3, device=device)
        target_points = torch.randn(1, num_points, 3, device=device)
        
        # 创建随机网格
        num_vertices = 100
        num_faces = 200
        vertices = torch.randn(1, num_vertices, 3, device=device)
        faces = torch.randint(0, num_vertices, (1, num_faces, 3), device=device)
        
        # 计算Chamfer距离
        cd_loss = chamfer_distance(source_points, target_points)
        print(f"Chamfer距离: {cd_loss.item()}")
        
        # 计算Laplacian正则化损失
        lap_loss = laplace_regularizer_const(vertices, faces)
        print(f"Laplacian正则化损失: {lap_loss.item()}")
        
        print("✓ 损失函数测试通过")
        return True
    except Exception as e:
        print(f"✗ 损失函数测试失败: {str(e)}")
        return False

@hydra.main(version_base=None, config_path="../configs", config_name="config_schema")
def main(config: DictConfig):
    """
    主函数,运行所有测试
    """
    print_separator("开始模块正确性测试")
    print(f"配置信息:\n{OmegaConf.to_yaml(config)}")
    
    # 运行测试
    tests = [
        ("数据加载", test_data_loading),
        ("四面体网格加载", test_tetrahedral_grid_loading),
        ("模型前向传播", test_model_forward),
        ("Marching Tetrahedra算法", test_marching_tetrahedra),
        ("点云采样", test_point_sampling),
        ("损失函数", test_loss_functions)
    ]
    
    results = {}
    for name, test_func in tests:
        try:
            result = test_func(config)
            results[name] = result
        except Exception as e:
            print(f"测试 {name} 发生异常: {str(e)}")
            results[name] = False
    
    # 打印测试结果摘要
    print_separator("测试结果摘要")
    for name, result in results.items():
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{name}: {status}")
    
    # 总结
    if all(results.values()):
        print("\n所有测试都通过了!项目各模块运行正常.")
    else:
        print("\n有些测试失败了,请检查上面的错误信息.")

if __name__ == "__main__":
    main()