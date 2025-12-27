"""模型训练模块"""

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import time
import numpy as np
from pathlib import Path
import json

class Trainer:
    """模型训练器"""
    
    def __init__(self, model, config, device=None, activation_name=None):
        self.model = model
        self.config = config
        self.activation_name = activation_name
        
        # 设备设置
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
            
        self.model.to(self.device)
        
        # 损失函数和优化器
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            model.parameters(), 
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            factor=0.5, 
            patience=3,
        )
        
        # 训练历史记录
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rate': [],
            'epoch_time': [],
            'gradient_stats': []
        }
        
        # 添加缺失的属性
        self.epoch = 0
        self.best_val_accuracy = 0.0
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        
        print(f"训练器初始化完成，设备: {self.device}")
    
    def train_epoch(self, train_loader):
        """训练一个epoch"""
        self.model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc='Training', leave=False)
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪（防止梯度爆炸）
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # 记录梯度信息（每10个batch记录一次）
            if batch_idx % 10 == 0:
                grad_info = self.model.get_gradient_info()
                self.history['gradient_stats'].append({
                    'batch': batch_idx,
                    'grad_info': grad_info
                })
            
            # 参数更新
            self.optimizer.step()
            
            # 统计
            epoch_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
            # 更新进度条
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
        
        epoch_loss /= len(train_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate(self, val_loader):
        """验证模型"""
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                val_loss += self.criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        
        val_loss /= len(val_loader)
        val_acc = 100. * correct / total
        
        return val_loss, val_acc
    
    def train(self, train_loader, val_loader, epochs=None):
        """完整训练过程"""
        if epochs is None:
            epochs = self.config.EPOCHS
        
        print(f"开始训练 {epochs} 个epochs")
        print("-" * 60)
        
        best_val_acc = 0.0
        best_epoch = 0
        
        for epoch in range(epochs):
            epoch_start_time = time.time()
            
            # 训练一个epoch
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # 验证
            val_loss, val_acc = self.validate(val_loader)
            
            # 学习率调整
            self.scheduler.step(val_loss)
            
            # 记录历史
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['learning_rate'].append(
                self.optimizer.param_groups[0]['lr']
            )
            self.history['epoch_time'].append(time.time() - epoch_start_time)
            
            # 更新属性
            self.epoch = epoch + 1
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)
            
            # 保存最佳模型
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch + 1
                self.best_val_accuracy = best_val_acc
                
                # 确保目录存在
                Path(self.config.MODEL_SAVE_DIR).mkdir(exist_ok=True)
                
                # 保存模型
                model_path = f"{self.config.MODEL_SAVE_DIR}/best_model_{self.activation_name}.pth"
                self.save_model(model_path)
                print(f"✓ 最佳模型已保存: {model_path}")
            
            # 打印进度
            print(f'Epoch {epoch+1:03d}/{epochs:03d}: '
                  f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | '
                  f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}% | '
                  f'LR: {self.history["learning_rate"][-1]:.6f}')
            
            # 早停检查（如果连续5个epoch验证损失没有改善）
            if len(self.history['val_loss']) >= 5:
                recent_losses = self.history['val_loss'][-5:]
                if all(recent_losses[i] >= recent_losses[i-1] 
                       for i in range(1, 5)):
                    print("检测到验证损失不再下降，提前停止训练")
                    break
        
        print(f"\n训练完成！最佳验证准确率: {best_val_acc:.2f}% (epoch {best_epoch})")
        return self.history
    
    def save_model(self, path):
        """保存模型"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'epoch': self.epoch,
            'best_val_accuracy': self.best_val_accuracy,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies
        }, path)
      
    def load_model(self, path):
        """加载模型"""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.epoch = checkpoint['epoch']
        self.best_val_accuracy = checkpoint['best_val_accuracy']
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        self.val_accuracies = checkpoint['val_accuracies']

class ExperimentRunner:
    """实验运行器：管理多个模型的训练"""
    
    def __init__(self, config):
        self.config = config
        self.results = {}
        
    def run_single_experiment(self, activation_name, train_loader, val_loader):
        """运行单个激活函数的实验"""
        print(f"\n{'='*60}")
        print(f"开始实验: {activation_name} 激活函数")
        print(f"{'='*60}")
        
        # 创建模型
        from models import ActivationFactory
        model = ActivationFactory.create_model(activation_name, self.config)
        
        # 创建训练器（传入激活函数名称）
        trainer = Trainer(model, self.config, activation_name=activation_name)
        
        # 训练
        history = trainer.train(train_loader, val_loader)
        
        # 保存结果
        self.results[activation_name] = {
            'history': history,
            'best_val_acc': max(history['val_acc']),
            'best_epoch': np.argmax(history['val_acc']) + 1,
            'final_train_acc': history['train_acc'][-1],
            'final_val_acc': history['val_acc'][-1],
            'total_training_time': sum(history['epoch_time']),
            'gradient_stats': history['gradient_stats']
        }
        
        # 保存详细结果
        self.save_experiment_result(activation_name)
        
        return self.results[activation_name]
    
    def run_all_experiments(self, train_loader, val_loader):
        """运行所有激活函数的实验"""
        all_results = {}
        
        for activation_name in self.config.ACTIVATIONS.keys():
            result = self.run_single_experiment(
                activation_name, train_loader, val_loader
            )
            all_results[activation_name] = result
        
        # 保存总体结果
        self.save_all_results(all_results)
        
        return all_results
    
    def save_experiment_result(self, activation_name):
        """保存单个实验结果"""
        Path(self.config.RESULT_SAVE_DIR).mkdir(exist_ok=True)
        
        result = self.results[activation_name]
        filename = f"result_{activation_name}_{time.strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(f"{self.config.RESULT_SAVE_DIR}/{filename}", 'w') as f:
            json.dump(result, f, indent=2, default=str)
    
    def save_all_results(self, all_results):
        """保存所有实验结果"""
        summary = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'config': self.config.__dict__,
            'results': {}
        }
        
        for act_name, result in all_results.items():
            summary['results'][act_name] = {
                'best_val_acc': result['best_val_acc'],
                'best_epoch': result['best_epoch'],
                'final_val_acc': result['final_val_acc'],
                'total_training_time': result['total_training_time']
            }
        
        filename = f"experiment_summary_{time.strftime('%Y%m%d_%H%M%S')}.json"
        with open(f"{self.config.RESULT_SAVE_DIR}/{filename}", 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n实验总结已保存到: {self.config.RESULT_SAVE_DIR}/{filename}")
        
        # 打印总结表格
        self.print_summary_table(summary)
    
    def print_summary_table(self, summary):
        """打印结果总结表格"""
        print("\n" + "="*80)
        print("实验总结")
        print("="*80)
        print(f"{'激活函数':<12} {'最佳验证准确率':<15} {'最佳epoch':<10} {'最终验证准确率':<15} {'训练时间(s)':<12}")
        print("-"*80)
        
        for act_name, result in summary['results'].items():
            print(f"{act_name:<12} {result['best_val_acc']:<15.2f} {result['best_epoch']:<10} "
                  f"{result['final_val_acc']:<15.2f} {result['total_training_time']:<12.2f}")


def main():
    """测试训练模块"""
    from config import Config
    from data_loader import DataLoaderManager
    
    config = Config()
    data_manager = DataLoaderManager(config)
    train_loader, val_loader, _ = data_manager.get_dataloaders()
    
    runner = ExperimentRunner(config)
    results = runner.run_single_experiment('ReLU', train_loader, val_loader)
    
    return results


if __name__ == "__main__":
    main()