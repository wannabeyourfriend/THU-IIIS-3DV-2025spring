import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import open3d as o3d
from PIL import Image
import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from torchvision import transforms
import logging

from model import PointCloudAutoEncoder
from dataset import CubeDataset

# Get logger
log = logging.getLogger(__name__)

def visualize_point_cloud(points, title, filename=None, color=None, realistic=False, sphere_radius=0.01):
    """
    Visualize point cloud
    
    Args:
        points: Point cloud coordinates, shape (N, 3)
        title: Plot title
        filename: Save filename, if None don't save
        color: Point color
        realistic: Whether to use realistic rendering with spheres
        sphere_radius: Radius of spheres for realistic rendering
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    if color is None:
        color = 'b'
    
    if realistic:
        # Use realistic rendering with spheres
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        
        # Calculate the number of points to render based on performance considerations
        max_spheres = 1000  # Maximum number of spheres to render for performance
        if len(points) > max_spheres:
            # Subsample points if there are too many
            indices = np.random.choice(len(points), max_spheres, replace=False)
            render_points = points[indices]
            log.info(f"Subsampling {len(points)} points to {max_spheres} for realistic rendering")
        else:
            render_points = points
        
        # Create spheres for each point
        for p in render_points:
            u = np.linspace(0, 2 * np.pi, 10)
            v = np.linspace(0, np.pi, 10)
            x = p[0] + sphere_radius * np.outer(np.cos(u), np.sin(v))
            y = p[1] + sphere_radius * np.outer(np.sin(u), np.sin(v))
            z = p[2] + sphere_radius * np.outer(np.ones(np.size(u)), np.cos(v))
            ax.plot_surface(x, y, z, color=color, alpha=0.8)
    else:
        # Use standard point cloud rendering
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=color, s=1)
    
    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # Set equal axis scale
    max_range = np.max([
        np.max(points[:, 0]) - np.min(points[:, 0]),
        np.max(points[:, 1]) - np.min(points[:, 1]),
        np.max(points[:, 2]) - np.min(points[:, 2])
    ])
    
    mid_x = (np.max(points[:, 0]) + np.min(points[:, 0])) * 0.5
    mid_y = (np.max(points[:, 1]) + np.min(points[:, 1])) * 0.5
    mid_z = (np.max(points[:, 2]) + np.min(points[:, 2])) * 0.5
    
    ax.set_xlim(mid_x - max_range * 0.5, mid_x + max_range * 0.5)
    ax.set_ylim(mid_y - max_range * 0.5, mid_y + max_range * 0.5)
    ax.set_zlim(mid_z - max_range * 0.5, mid_z + max_range * 0.5)
    
    if filename is not None:
        plt.savefig(filename)
        log.info(f"Saved image to {filename}")
    
    return fig, ax

def visualize_comparison(image, gt_points, pred_points, index, output_dir, realistic=False, sphere_radius=0.01):
    """
    Visualize comparison between input image, ground truth and predicted point clouds
    
    Args:
        image: Input image
        gt_points: Ground truth point cloud
        pred_points: Predicted point cloud
        index: Sample index
        output_dir: Output directory
        realistic: Whether to use realistic rendering with spheres
        sphere_radius: Radius of spheres for realistic rendering
    """
    # Create figure
    fig = plt.figure(figsize=(15, 5))
    
    # Display input image
    ax1 = fig.add_subplot(131)
    # Convert tensor to numpy and adjust channel order
    img_np = image.cpu().numpy().transpose(1, 2, 0)
    # Denormalize
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_np = std * img_np + mean
    img_np = np.clip(img_np, 0, 1)
    ax1.imshow(img_np)
    ax1.set_title("Input Image")
    ax1.axis('off')
    
    # Display ground truth point cloud
    ax2 = fig.add_subplot(132, projection='3d')
    
    if realistic:
        # Use realistic rendering with spheres
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        
        # Calculate the number of points to render based on performance considerations
        max_spheres = 500  # Maximum number of spheres to render for performance
        if len(gt_points) > max_spheres:
            # Subsample points if there are too many
            indices = np.random.choice(len(gt_points), max_spheres, replace=False)
            render_points = gt_points[indices]
        else:
            render_points = gt_points
        
        # Create spheres for each point
        for p in render_points:
            u = np.linspace(0, 2 * np.pi, 8)
            v = np.linspace(0, np.pi, 8)
            x = p[0] + sphere_radius * np.outer(np.cos(u), np.sin(v))
            y = p[1] + sphere_radius * np.outer(np.sin(u), np.sin(v))
            z = p[2] + sphere_radius * np.outer(np.ones(np.size(u)), np.cos(v))
            ax2.plot_surface(x, y, z, color='r', alpha=0.8)
    else:
        ax2.scatter(gt_points[:, 0], gt_points[:, 1], gt_points[:, 2], c='r', s=1)
    
    ax2.set_title("Ground Truth")
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    
    # Display predicted point cloud
    ax3 = fig.add_subplot(133, projection='3d')
    
    if realistic:
        # Use realistic rendering with spheres
        # Calculate the number of points to render based on performance considerations
        max_spheres = 500  # Maximum number of spheres to render for performance
        if len(pred_points) > max_spheres:
            # Subsample points if there are too many
            indices = np.random.choice(len(pred_points), max_spheres, replace=False)
            render_points = pred_points[indices]
        else:
            render_points = pred_points
        
        # Create spheres for each point
        for p in render_points:
            u = np.linspace(0, 2 * np.pi, 8)
            v = np.linspace(0, np.pi, 8)
            x = p[0] + sphere_radius * np.outer(np.cos(u), np.sin(v))
            y = p[1] + sphere_radius * np.outer(np.sin(u), np.sin(v))
            z = p[2] + sphere_radius * np.outer(np.ones(np.size(u)), np.cos(v))
            ax3.plot_surface(x, y, z, color='b', alpha=0.8)
    else:
        ax3.scatter(pred_points[:, 0], pred_points[:, 1], pred_points[:, 2], c='b', s=1)
    
    ax3.set_title("Prediction")
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')
    
    # Set same view angle for both point clouds
    for ax in [ax2, ax3]:
        max_range = np.max([
            np.max(gt_points[:, 0]) - np.min(gt_points[:, 0]),
            np.max(gt_points[:, 1]) - np.min(gt_points[:, 1]),
            np.max(gt_points[:, 2]) - np.min(gt_points[:, 2])
        ])
        
        mid_x = (np.max(gt_points[:, 0]) + np.min(gt_points[:, 0])) * 0.5
        mid_y = (np.max(gt_points[:, 1]) + np.min(gt_points[:, 1])) * 0.5
        mid_z = (np.max(gt_points[:, 2]) + np.min(gt_points[:, 2])) * 0.5
        
        ax.set_xlim(mid_x - max_range * 0.5, mid_x + max_range * 0.5)
        ax.set_ylim(mid_y - max_range * 0.5, mid_y + max_range * 0.5)
        ax.set_zlim(mid_z - max_range * 0.5, mid_z + max_range * 0.5)
        ax.view_init(elev=30, azim=45)
    
    plt.tight_layout()
    render_type = "realistic" if realistic else "standard"
    filename = os.path.join(output_dir, f"comparison_{index}_{render_type}.png")
    plt.savefig(filename)
    plt.close()
    log.info(f"Saved comparison image to {filename}")

def save_point_cloud_ply(points, filename):
    """
    Save point cloud to PLY file
    
    Args:
        points: Point cloud coordinates, shape (N, 3)
        filename: Save filename
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.io.write_point_cloud(filename, pcd)
    log.info(f"Saved point cloud to {filename}")

def create_rotating_animation(points, output_file, color=None, realistic=False, sphere_radius=0.01):
    """
    Create rotating animation of point cloud
    
    Args:
        points: Point cloud coordinates, shape (N, 3)
        output_file: Output filename
        color: Point color
        realistic: Whether to use realistic rendering with spheres
        sphere_radius: Radius of spheres for realistic rendering
    """
    import matplotlib.animation as animation
    
    # Create figure
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    if color is None:
        color = 'b'
    
    if realistic:
        # For realistic rendering, we'll create a new plot for each frame
        # Subsample points for better performance
        max_spheres = 300  # Maximum number of spheres to render for animation
        if len(points) > max_spheres:
            indices = np.random.choice(len(points), max_spheres, replace=False)
            render_points = points[indices]
            log.info(f"Subsampling {len(points)} points to {max_spheres} for realistic animation")
        else:
            render_points = points
        
        # Pre-compute sphere coordinates
        spheres = []
        for p in render_points:
            u = np.linspace(0, 2 * np.pi, 8)
            v = np.linspace(0, np.pi, 8)
            x = p[0] + sphere_radius * np.outer(np.cos(u), np.sin(v))
            y = p[1] + sphere_radius * np.outer(np.sin(u), np.sin(v))
            z = p[2] + sphere_radius * np.outer(np.ones(np.size(u)), np.cos(v))
            spheres.append((x, y, z))
        
        surfaces = []
        for x, y, z in spheres:
            surf = ax.plot_surface(x, y, z, color=color, alpha=0.8)
            surfaces.append(surf)
    else:
        # Use standard point cloud rendering
        scatter = ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=color, s=1)
    
    # Set equal axis scale
    max_range = np.max([
        np.max(points[:, 0]) - np.min(points[:, 0]),
        np.max(points[:, 1]) - np.min(points[:, 1]),
        np.max(points[:, 2]) - np.min(points[:, 2])
    ])
    
    mid_x = (np.max(points[:, 0]) + np.min(points[:, 0])) * 0.5
    mid_y = (np.max(points[:, 1]) + np.min(points[:, 1])) * 0.5
    mid_z = (np.max(points[:, 2]) + np.min(points[:, 2])) * 0.5
    
    ax.set_xlim(mid_x - max_range * 0.5, mid_x + max_range * 0.5)
    ax.set_ylim(mid_y - max_range * 0.5, mid_y + max_range * 0.5)
    ax.set_zlim(mid_z - max_range * 0.5, mid_z + max_range * 0.5)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Point Cloud Rotation')
    
    # Define update function
    if realistic:
        def update(frame):
            ax.view_init(elev=30, azim=frame)
            return surfaces
    else:
        def update(frame):
            ax.view_init(elev=30, azim=frame)
            return scatter,
    
    # Create animation
    ani = animation.FuncAnimation(fig, update, frames=range(0, 360, 6), 
                                  interval=50, blit=True if not realistic else False)
    
    # Save animation
    ani.save(output_file, writer='pillow', fps=15)
    plt.close()
    log.info(f"Saved animation to {output_file}")

@hydra.main(config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    # Print configuration
    log.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log.info(f'Using device: {device}')
    
    # Data preprocessing
    transform = transforms.Compose([
        transforms.Resize((cfg.image_size, cfg.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=cfg.normalize.mean, std=cfg.normalize.std)
    ])
    
    # Convert dataset path to absolute path
    data_root = to_absolute_path(cfg.data.root_dir)
    
    # Load test dataset
    test_dataset = CubeDataset(
        root_dir=data_root, 
        split='test', 
        transform=transform, 
        num_points=cfg.model.num_points,
        clean=cfg.data.clean
    )
    log.info(f"Test dataset size: {len(test_dataset)}")
    
    # Create output directory
    output_dir = os.path.join(os.getcwd(), "visualization")
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model
    model_path = to_absolute_path(cfg.visualization.model_path)
    log.info(f"Loading model: {model_path}")
    
    model = PointCloudAutoEncoder(
        num_points=cfg.model.num_points, 
        feature_dim=cfg.model.feature_dim
    ).to(device)
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    log.info(f"Model loaded successfully, starting visualization...")
    
    # Choose number of samples to visualize
    num_samples = min(cfg.visualization.num_samples, len(test_dataset))
    
    # Check if realistic rendering is enabled
    realistic = cfg.visualization.get('realistic', False)
    sphere_radius = cfg.visualization.get('sphere_radius', 0.01)
    
    if realistic:
        log.info(f"Using realistic rendering with sphere radius: {sphere_radius}")
    
    # Generate visualization results
    with torch.no_grad():
        for i in range(num_samples):
            # Get sample
            image, gt_points = test_dataset[i]
            image = image.unsqueeze(0).to(device)  # Add batch dimension
            gt_points = gt_points.to(device)
            
            # Model inference
            pred_points = model(image).squeeze(0)  # Remove batch dimension
            
            # Convert to numpy arrays
            gt_np = gt_points.cpu().numpy()
            pred_np = pred_points.cpu().numpy()
            
            # Visualize comparison
            visualize_comparison(
                image.squeeze(0), 
                gt_np, 
                pred_np, 
                i, 
                output_dir,
                realistic=realistic,
                sphere_radius=sphere_radius
            )
            
            # Save point cloud files
            save_point_cloud_ply(
                gt_np, 
                os.path.join(output_dir, f"gt_{i}.ply")
            )
            save_point_cloud_ply(
                pred_np, 
                os.path.join(output_dir, f"pred_{i}.ply")
            )
            
            # Create rotating animation for first sample
            if i == 0:
                create_rotating_animation(
                    gt_np, 
                    os.path.join(output_dir, f"gt_rotating{'_realistic' if realistic else ''}.gif"), 
                    color='r',
                    realistic=realistic,
                    sphere_radius=sphere_radius
                )
                create_rotating_animation(
                    pred_np, 
                    os.path.join(output_dir, f"pred_rotating{'_realistic' if realistic else ''}.gif"), 
                    color='b',
                    realistic=realistic,
                    sphere_radius=sphere_radius
                )
    
    log.info(f"Visualization completed, results saved in {output_dir}")

if __name__ == '__main__':
    main()