import os
import torch
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
import logging

from loss import ChamferDistanceLoss, HausdorffDistanceLoss
from model import PointCloudAutoEncoder
from train import train_model, evaluate_model
from dataset import CubeDataset

# Get logger
log = logging.getLogger(__name__)

@hydra.main(config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    # Print configuration
    log.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")
    
    # Set random seed
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(cfg.seed)
        torch.cuda.manual_seed_all(cfg.seed)
        torch.backends.cudnn.deterministic = True
    
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
    
    # Load datasets
    train_dataset = CubeDataset(
        root_dir=data_root, 
        split='train', 
        transform=transform, 
        num_points=cfg.model.num_points,
        clean=cfg.data.clean
    )
    
    # Check dataset sizes and log information
    log.info(f"Data root path: {data_root}")
    log.info(f"Train dataset size: {len(train_dataset)}")
    if len(train_dataset) == 0:
        raise ValueError(f"Train dataset is empty! Please check the data path: {data_root}")
    
    val_dataset = CubeDataset(
        root_dir=data_root, 
        split='val', 
        transform=transform, 
        num_points=cfg.model.num_points,
        clean=cfg.data.clean
    )
    log.info(f"Validation dataset size: {len(val_dataset)}")
    
    test_dataset = CubeDataset(
        root_dir=data_root, 
        split='test', 
        transform=transform, 
        num_points=cfg.model.num_points,
        clean=cfg.data.clean
    )
    log.info(f"Test dataset size: {len(test_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=cfg.train.batch_size, 
        shuffle=True, 
        num_workers=cfg.train.num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=cfg.train.batch_size, 
        shuffle=False, 
        num_workers=cfg.train.num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=1, 
        shuffle=False, 
        num_workers=cfg.train.num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    # Create loss functions
    cd_loss = ChamferDistanceLoss().to(device)
    hd_loss = HausdorffDistanceLoss().to(device)
    
    # Train model with Chamfer Distance
    if cfg.train.use_cd:
        log.info("Training model with Chamfer Distance loss...")
        model_cd = PointCloudAutoEncoder(
            num_points=cfg.model.num_points, 
            feature_dim=cfg.model.feature_dim
        ).to(device)
        
        optimizer_cd = optim.Adam(
            model_cd.parameters(), 
            lr=cfg.train.lr,
            weight_decay=1e-4
        )
        
        scheduler_cd = optim.lr_scheduler.StepLR(
            optimizer_cd, 
            step_size=cfg.train.lr_step, 
            gamma=cfg.train.lr_gamma
        )
        
        train_losses_cd, val_losses_cd = train_model(
            model=model_cd,
            train_loader=train_loader,
            val_loader=val_loader,
            loss_fn=cd_loss,
            optimizer=optimizer_cd,
            scheduler=scheduler_cd,
            num_epochs=cfg.train.epochs,
            device=device,
            save_dir=os.path.join(os.getcwd(), "model_cd")
        )
        
        log.info("Evaluating model trained with Chamfer Distance...")
        test_loss_cd = evaluate_model(model_cd, test_loader, cd_loss, device=device)
        log.info(f'Final test loss (CD): {test_loss_cd:.6f}')
        
        # Save training curves
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses_cd, label='Train Loss')
        plt.plot(val_losses_cd, label='Validation Loss')
        plt.title('Chamfer Distance Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(os.path.join(os.getcwd(), "cd_loss_curve.png"))
        plt.close()
    
    # Train model with Hausdorff Distance
    if cfg.train.use_hd:
        log.info("Training model with Hausdorff Distance loss...")
        model_hd = PointCloudAutoEncoder(
            num_points=cfg.model.num_points, 
            feature_dim=cfg.model.feature_dim
        ).to(device)
        
        optimizer_hd = optim.Adam(
            model_hd.parameters(), 
            lr=cfg.train.lr
        )
        
        scheduler_hd = optim.lr_scheduler.StepLR(
            optimizer_hd, 
            step_size=cfg.train.lr_step, 
            gamma=cfg.train.lr_gamma
        )
        
        train_losses_hd, val_losses_hd = train_model(
            model=model_hd,
            train_loader=train_loader,
            val_loader=val_loader,
            loss_fn=hd_loss,
            optimizer=optimizer_hd,
            scheduler=scheduler_hd,
            num_epochs=cfg.train.epochs,
            device=device,
            save_dir=os.path.join(os.getcwd(), "model_hd")
        )
        
        log.info("Evaluating model trained with Hausdorff Distance...")
        test_loss_hd = evaluate_model(model_hd, test_loader, hd_loss, device=device)
        log.info(f'Final test loss (HD): {test_loss_hd:.6f}')
        
        # Save training curves
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses_hd, label='Train Loss')
        plt.plot(val_losses_hd, label='Validation Loss')
        plt.title('Hausdorff Distance Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(os.path.join(os.getcwd(), "hd_loss_curve.png"))
        plt.close()
    
    # If both losses are used, draw comparison plot
    if cfg.train.use_cd and cfg.train.use_hd:
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(train_losses_cd, label='Train Loss')
        plt.plot(val_losses_cd, label='Validation Loss')
        plt.title('Chamfer Distance Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(train_losses_hd, label='Train Loss')
        plt.plot(val_losses_hd, label='Validation Loss')
        plt.title('Hausdorff Distance Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(os.getcwd(), "loss_comparison.png"))
        plt.close()
        
        log.info(f'Final test loss comparison - CD: {test_loss_cd:.6f}, HD: {test_loss_hd:.6f}')

if __name__ == '__main__':
    main()
