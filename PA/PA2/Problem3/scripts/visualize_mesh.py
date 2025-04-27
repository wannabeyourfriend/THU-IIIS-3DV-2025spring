import numpy as np
import pyvista as pv
import os

def inspect_npz_file(npz_file):
    """检查npz文件的内容结构"""
    data = np.load(npz_file)
    print("\n文件内容结构:")
    print("=" * 50)
    for key in data.files:
        array = data[key]
        print(f"键名: {key}")
        print(f"形状: {array.shape}")
        print(f"数据类型: {array.dtype}")
        print("-" * 50)
    return data

def visualize_tetra_mesh(npz_file):
    # 首先检查文件内容
    data = inspect_npz_file(npz_file)
    
    # 获取点坐标和四面体网格
    # 注意：这里需要根据实际的文件结构来调整键名
    points = data[data.files[0]]
    tets = data[data.files[1]]
    
    # 如果tets不是[n, 4]，需要reshape
    if tets.shape[1] != 4:
        tets = tets.reshape(-1, 4)
    
    # 构造cells数组（每个四面体前面加一个4，表示四面体有4个顶点）
    cells = np.hstack([np.full((tets.shape[0], 1), 4), tets]).astype(np.int64)
    cells = cells.flatten()
    
    # 创建UnstructuredGrid
    grid = pv.UnstructuredGrid(cells, np.full(tets.shape[0], pv.CellType.TETRA), points)
    
    # 创建可视化窗口
    plotter = pv.Plotter()
    # 先画体
    plotter.add_mesh(grid, show_edges=False, color='lightblue', opacity=0.3)
    # 再画边
    plotter.add_mesh(grid.extract_all_edges(), color='black', line_width=1)
    plotter.show_axes()
    plotter.show()

def main():
    # 设置tetra网格文件所在的目录
    tets_dir = 'data/tets'
    
    # 获取所有tetra网格文件
    npz_files = [f for f in os.listdir(tets_dir) if f.endswith('_compress.npz')]
    
    if not npz_files:
        print("未找到tetra网格文件！")
        return
    
    print("可用的tetra网格文件：")
    for i, file in enumerate(npz_files):
        print(f"{i+1}. {file}")
    
    # 让用户选择要可视化的文件
    choice = int(input("请选择要可视化的文件编号（1-{}）: ".format(len(npz_files)))) - 1
    
    if 0 <= choice < len(npz_files):
        file_path = os.path.join(tets_dir, npz_files[choice])
        print(f"正在可视化: {npz_files[choice]}")
        visualize_tetra_mesh(file_path)
    else:
        print("无效的选择！")

if __name__ == "__main__":
    main() 