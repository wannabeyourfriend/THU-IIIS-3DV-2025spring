import os
import torch
from tqdm import tqdm
import logging

log = logging.getLogger(__name__)

def train_model(model, train_loader, val_loader, loss_fn, optimizer, scheduler, num_epochs, device, save_dir=None):
    """
    训练模型
    
    Args:
        model: 模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        loss_fn: 损失函数
        optimizer: 优化器
        scheduler: 学习率调度器
        num_epochs: 训练轮数
        device: 设备
        save_dir: 模型保存目录
    
    Returns:
        train_losses: 训练损失列表
        val_losses: 验证损失列表
    """
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    # 创建保存目录
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        
        for images, point_clouds in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"):
            images = images.to(device)
            point_clouds = point_clouds.to(device)
            
            # 前向传播
            outputs = model(images)
            
            # 计算损失
            loss = loss_fn(outputs, point_clouds)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # 计算平均训练损失
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for images, point_clouds in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"):
                images = images.to(device)
                point_clouds = point_clouds.to(device)
                
                # 前向传播
                outputs = model(images)
                
                # 计算损失
                loss = loss_fn(outputs, point_clouds)
                val_loss += loss.item()
        
        # 计算平均验证损失
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        # 更新学习率
        scheduler.step()
        
        # 打印进度
        log.info(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
        
        # 保存最佳模型
        if save_dir is not None and avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_val_loss,
            }, os.path.join(save_dir, 'best_model.pth'))
            log.info(f"Save best model, validation loss: {best_val_loss:.6f}")
    
    # 保存最终模型
    if save_dir is not None:
        torch.save({
            'epoch': num_epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_val_loss,
        }, os.path.join(save_dir, 'final_model.pth'))
        log.info(f"Save final model, validation loss: {avg_val_loss:.6f}")
    
    return train_losses, val_losses

def evaluate_model(model, test_loader, loss_fn, device):
    """
    评估模型
    
    Args:
        model: 模型
        test_loader: 测试数据加载器
        loss_fn: 损失函数
        device: 设备
    
    Returns:
        avg_test_loss: 平均测试损失
    """
    model.eval()
    test_loss = 0.0
    
    with torch.no_grad():
        for images, point_clouds in tqdm(test_loader, desc="Testing"):
            images = images.to(device)
            point_clouds = point_clouds.to(device)
            
            # 前向传播
            outputs = model(images)
            
            # 计算损失
            loss = loss_fn(outputs, point_clouds)
            test_loss += loss.item()
    
    # 计算平均测试损失
    avg_test_loss = test_loss / len(test_loader)
    
    return avg_test_loss
