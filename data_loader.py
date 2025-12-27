"""数据加载与预处理模块"""

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import numpy as np
from config import Config

class DataLoaderManager:
    def __init__(self, config):
        self.config = config
        self.set_seed()
        
    def set_seed(self):
        """设置随机种子确保可重复性"""
        torch.manual_seed(self.config.SEED)
        np.random.seed(self.config.SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.SEED)
    
    def get_transforms(self):
        """定义数据转换"""
        if self.config.DATASET_NAME == 'MNIST':
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
        elif self.config.DATASET_NAME == 'CIFAR10':
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), 
                                   (0.2470, 0.2435, 0.2616))
            ])
        else:
            raise ValueError(f"Unsupported dataset: {self.config.DATASET_NAME}")
        return transform
    
    def load_datasets(self):
        """加载训练和测试数据集"""
        transform = self.get_transforms()
        
        if self.config.DATASET_NAME == 'MNIST':
            # 训练集
            train_dataset = torchvision.datasets.MNIST(
                root=self.config.DATA_ROOT,
                train=True,
                download=True,
                transform=transform
            )
            
            # 划分部分数据作为验证集（10%）
            train_size = int(0.9 * len(train_dataset))
            val_size = len(train_dataset) - train_size
            train_dataset, val_dataset = random_split(
                train_dataset, [train_size, val_size]
            )
            
            # 测试集
            test_dataset = torchvision.datasets.MNIST(
                root=self.config.DATA_ROOT,
                train=False,
                download=True,
                transform=transform
            )
            
        elif self.config.DATASET_NAME == 'CIFAR10':
            # 训练集
            train_dataset = torchvision.datasets.CIFAR10(
                root=self.config.DATA_ROOT,
                train=True,
                download=True,
                transform=transform
            )
            
            # 验证集划分
            train_size = int(0.9 * len(train_dataset))
            val_size = len(train_dataset) - train_size
            train_dataset, val_dataset = random_split(
                train_dataset, [train_size, val_size]
            )
            
            # 测试集
            test_dataset = torchvision.datasets.CIFAR10(
                root=self.config.DATA_ROOT,
                train=False,
                download=True,
                transform=transform
            )
        
        return train_dataset, val_dataset, test_dataset
    
    def get_dataloaders(self):
        """创建数据加载器"""
        train_dataset, val_dataset, test_dataset = self.load_datasets()
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=True,
            num_workers=2,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.TEST_BATCH_SIZE,
            shuffle=False,
            num_workers=2
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.TEST_BATCH_SIZE,
            shuffle=False,
            num_workers=2
        )
        
        print(f"数据集加载完成:")
        print(f"  训练集: {len(train_dataset)} 样本")
        print(f"  验证集: {len(val_dataset)} 样本")
        print(f"  测试集: {len(test_dataset)} 样本")
        
        return train_loader, val_loader, test_loader