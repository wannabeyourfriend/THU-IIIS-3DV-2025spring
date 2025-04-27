import os
import torch
from tqdm import tqdm
import logging

log = logging.getLogger(__name__)

def train_model(model, train_loader, val_loader, loss_fn, optimizer, scheduler, num_epochs, device, save_dir=None):
    """
    Train the model
    
    Args:
        model: Model instance
        train_loader: Training data loader
        val_loader: Validation data loader
        loss_fn: Loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        num_epochs: Number of epochs
        device: Device (cpu or cuda)
        save_dir: Directory to save models
    
    Returns:
        train_losses: List of training losses
        val_losses: List of validation losses
    """
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    # Create save directory
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for images, point_clouds in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"):
            images = images.to(device)
            point_clouds = point_clouds.to(device)
            
            # Forward pass
            outputs = model(images)
            
            # Compute loss
            loss = loss_fn(outputs, point_clouds)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Compute average training loss
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for images, point_clouds in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"):
                images = images.to(device)
                point_clouds = point_clouds.to(device)
                
                # Forward pass
                outputs = model(images)
                
                # Compute loss
                loss = loss_fn(outputs, point_clouds)
                val_loss += loss.item()
        
        # Compute average validation loss
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        # Update learning rate
        scheduler.step()
        
        # Print progress
        log.info(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
        
        # Save best model
        if save_dir is not None and avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_val_loss,
            }, os.path.join(save_dir, 'best_model.pth'))
            log.info(f"Saved best model, validation loss: {best_val_loss:.6f}")
    
    # Save final model
    if save_dir is not None:
        torch.save({
            'epoch': num_epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_val_loss,
        }, os.path.join(save_dir, 'final_model.pth'))
        log.info(f"Saved final model, validation loss: {avg_val_loss:.6f}")
    
    return train_losses, val_losses

def evaluate_model(model, test_loader, loss_fn, device):
    """
    Evaluate the model
    
    Args:
        model: Model instance
        test_loader: Test data loader
        loss_fn: Loss function
        device: Device (cpu or cuda)
    
    Returns:
        avg_test_loss: Average test loss
    """
    model.eval()
    test_loss = 0.0
    
    with torch.no_grad():
        for images, point_clouds in tqdm(test_loader, desc="Testing"):
            images = images.to(device)
            point_clouds = point_clouds.to(device)
            
            # Forward pass
            outputs = model(images)
            
            # Compute loss
            loss = loss_fn(outputs, point_clouds)
            test_loss += loss.item()
    
    # Compute average test loss
    avg_test_loss = test_loss / len(test_loader)
    
    return avg_test_loss
