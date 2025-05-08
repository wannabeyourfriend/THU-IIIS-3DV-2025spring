import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from model import PointCloudAutoEncoder
from dataset import CubeDataset
from loss import ChamferDistanceLoss, HausdorffDistanceLoss

def test_dataset():
    """Test if dataset loading is correct"""
    print("Testing dataset loading...")
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load small amount of data for testing
    test_dataset = CubeDataset(
        root_dir=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data'),
        split='test',
        transform=transform,
        num_points=1024,
        clean=True
    )
    
    if len(test_dataset) == 0:
        print("Error: Dataset is empty!")
        return False
    
    # Check data format
    image, point_cloud = test_dataset[0]
    print(f"Image shape: {image.shape}")
    print(f"Point cloud shape: {point_cloud.shape}")
    
    # Validate shapes
    assert image.shape == (3, 224, 224), "Unexpected image shape"
    assert point_cloud.shape == (1024, 3), "Unexpected point cloud shape"
    
    # Check data loader
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, num_workers=0)
    batch = next(iter(test_loader))
    assert len(batch) == 2, "Incorrect batch format from data loader"
    
    print("Dataset test passed!")
    return True

def test_model():
    """Test if model forward pass is correct"""
    print("Testing model forward pass...")
    
    # Create a small model for testing
    model = PointCloudAutoEncoder(num_points=1024, feature_dim=128)
    
    # Create random input
    batch_size = 2
    dummy_input = torch.randn(batch_size, 3, 224, 224)
    
    # Forward pass
    try:
        output = model(dummy_input)
        print(f"Model output shape: {output.shape}")
        
        # Validate output shape
        assert output.shape == (batch_size, 1024, 3), "Model output shape doesn't match expected shape"
        print("Model test passed!")
        return True
    except Exception as e:
        print(f"Model test failed: {e}")
        return False

def test_loss_functions():
    """Test if loss functions are correct"""
    print("Testing loss functions...")
    
    # Create random point cloud
    batch_size = 2
    num_points = 1024
    pred_points = torch.randn(batch_size, num_points, 3)
    gt_points = torch.randn(batch_size, num_points, 3)
    
    # Test Chamfer distance
    cd_loss = ChamferDistanceLoss()
    try:
        cd_value = cd_loss(pred_points, gt_points)
        print(f"Chamfer distance: {cd_value.item()}")
        assert cd_value.item() > 0, "Chamfer distance should be positive"
    except Exception as e:
        print(f"Chamfer distance test failed: {e}")
        return False

    # Test Hausdorff distance
    hd_loss = HausdorffDistanceLoss()
    try:
        hd_value = hd_loss(pred_points, gt_points)
        print(f"Hausdorff distance: {hd_value.item()}")
        assert hd_value.item() > 0, "Hausdorff distance should be positive"
    except Exception as e:
        print(f"Hausdorff distance test failed: {e}")
        return False
    
    # Test same point cloud loss
    try:
        same_cd = cd_loss(gt_points, gt_points)
        same_hd = hd_loss(gt_points, gt_points)
        print(f"Same point cloud's Chamfer distance: {same_cd.item()}")
        print(f"Same point cloud's Hausdorff distance: {same_hd.item()}")
        assert same_cd.item() < 1e-5, "Same point cloud's Chamfer distance should be close to 0"
        assert same_hd.item() < 1e-5, "Same point cloud's Hausdorff distance should be close to 0"
    except Exception as e:
        print(f"Same point cloud loss test failed: {e}")
        return False
    
    print("Loss function test passed!")
    return True

def test_training_step():
    """Test if single training step is correct"""
    print("Testing training step...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create small model
    model = PointCloudAutoEncoder(num_points=1024, feature_dim=128).to(device)
    
    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Create loss function
    cd_loss = ChamferDistanceLoss()
    
    # Create random input
    batch_size = 2
    dummy_input = torch.randn(batch_size, 3, 224, 224).to(device)
    dummy_target = torch.randn(batch_size, 1024, 3).to(device)
    
    # Execute single training step
    try:
        model.train()
        optimizer.zero_grad()
        
        # Forward pass
        output = model(dummy_input)
        
        # Compute loss
        loss = cd_loss(output, dummy_target)
        
        # Backward pass
        loss.backward()
        
        # Parameter update
        optimizer.step()
        
        print(f"Training loss: {loss.item()}")
        print("Training step test passed!")
        return True
    except Exception as e:
        print(f"Training step test failed: {e}")
        return False

def test_full_pipeline():
    """Test small batch training of complete pipeline"""
    print("Testing complete pipeline...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Set random seed to ensure reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Data preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load small batch data
    try:
        dataset = CubeDataset(
            root_dir=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data'),
            split='test',
            transform=transform,
            num_points=1024,
            clean=True
        )
        
        # Only use small amount of data for testing
        subset_size = min(10, len(dataset))
        indices = list(range(subset_size))
        subset = torch.utils.data.Subset(dataset, indices)
        
        dataloader = DataLoader(subset, batch_size=2, shuffle=True, num_workers=0)
        
        # Create model
        model = PointCloudAutoEncoder(num_points=1024, feature_dim=128).to(device)
        
        # Create optimizer and loss function
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        cd_loss = ChamferDistanceLoss()
        
        # Execute small batch training
        num_epochs = 2
        for epoch in range(num_epochs):
            model.train()
            epoch_loss = 0.0
            
            for images, point_clouds in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
                images = images.to(device)
                point_clouds = point_clouds.to(device)
                
                # Forward pass
                outputs = model(images)
                
                # Compute loss
                loss = cd_loss(outputs, point_clouds)
                
                # Backward pass and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(dataloader)
            print(f"Epoch {epoch+1}/{num_epochs}, Average loss: {avg_loss:.6f}")
        
        # Test inference
        model.eval()
        with torch.no_grad():
            test_images, test_points = next(iter(dataloader))
            test_images = test_images.to(device)
            test_points = test_points.to(device)
            
            # Forward pass
            outputs = model(test_images)
            
            # Compute test loss
            test_loss = cd_loss(outputs, test_points)
            print(f"Test loss: {test_loss.item():.6f}")
            
            # Visualize results
            output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'test_results')
            os.makedirs(output_dir, exist_ok=True)
            
            # Save first sample's prediction and ground truth point clouds
            pred_pc = outputs[0].cpu().numpy()
            gt_pc = test_points[0].cpu().numpy()
            
            fig = plt.figure(figsize=(12, 6))
            
            # Predicted point cloud
            ax1 = fig.add_subplot(121, projection='3d')
            ax1.scatter(pred_pc[:, 0], pred_pc[:, 1], pred_pc[:, 2], s=1, c='b')
            ax1.set_title('Predicted Point Cloud')
            ax1.set_xlabel('X')
            ax1.set_ylabel('Y')
            ax1.set_zlabel('Z')
            
            # Ground truth point cloud
            ax2 = fig.add_subplot(122, projection='3d')
            ax2.scatter(gt_pc[:, 0], gt_pc[:, 1], gt_pc[:, 2], s=1, c='r')
            ax2.set_title('Ground Truth Point Cloud')
            ax2.set_xlabel('X')
            ax2.set_ylabel('Y')
            ax2.set_zlabel('Z')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'test_result.png'))
            plt.close()
            
            print(f"Visualization results saved to {output_dir}")
        
        print("Complete pipeline test passed!")
        return True
    except Exception as e:
        print(f"Complete pipeline test failed: {e}")
        return False

if __name__ == "__main__":
    print("Starting pipeline tests...")
    
    # Test each module
    dataset_ok = test_dataset()
    model_ok = test_model()
    loss_ok = test_loss_functions()
    training_ok = test_training_step()
    
    # If all module tests pass, test complete pipeline
    if dataset_ok and model_ok and loss_ok and training_ok:
        pipeline_ok = test_full_pipeline()
        
        if pipeline_ok:
            print("\nAll tests passed! Pipeline is working correctly.")
        else:
            print("\nComplete pipeline test failed, but individual module tests passed.")
    else:
        print("\nSome module tests failed. Please fix the issues above.")