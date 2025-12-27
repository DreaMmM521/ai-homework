"""模型定义模块"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import Config

class ActivationMLP(nn.Module):
    """带有可配置激活函数的小型MLP"""
    
    def __init__(self, activation_fn_name, config):
        super(ActivationMLP, self).__init__()
        self.config = config
        self.activation_fn_name = activation_fn_name
        
        # 动态获取激活函数
        self.activation_fn = self._get_activation_fn(activation_fn_name)
        
        # 构建网络层
        self.layers = nn.ModuleList()
        sizes = [config.INPUT_SIZE] + config.HIDDEN_SIZES
        
        for i in range(len(sizes)-1):
            self.layers.append(nn.Linear(sizes[i], sizes[i+1]))
            if i < len(sizes)-2:  # 不在输出层前加激活函数
                self.layers.append(self.activation_fn())
        
        self.output_layer = nn.Linear(config.HIDDEN_SIZES[-1], config.OUTPUT_SIZE)
        
        # 初始化权重
        self._initialize_weights()
    
    def _get_activation_fn(self, name):
        """根据名称获取激活函数类"""
        if name == 'ReLU':
            return nn.ReLU
        elif name == 'Sigmoid':
            return nn.Sigmoid
        elif name == 'Tanh':
            return nn.Tanh
        elif name == 'LeakyReLU':
            return lambda: nn.LeakyReLU(self.config.LEAKY_RELU_NEGATIVE_SLOPE)
        else:
            raise ValueError(f"不支持的激活函数: {name}")
    
    def _initialize_weights(self):
        """Xavier初始化"""
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
        
        nn.init.xavier_uniform_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)
    
    def forward(self, x):
        """前向传播"""
        x = x.view(x.size(0), -1)  # 展平输入
        
        for layer in self.layers:
            x = layer(x)
        
        x = self.output_layer(x)
        return x
    
    def get_gradient_info(self):
        """获取梯度信息（用于分析）"""
        gradient_info = {}
        for name, param in self.named_parameters():
            if param.grad is not None:
                gradient_info[name] = {
                    'mean': param.grad.mean().item(),
                    'std': param.grad.std().item(),
                    'min': param.grad.min().item(),
                    'max': param.grad.max().item()
                }
        return gradient_info


class ActivationFactory:
    """激活函数模型工厂"""
    
    @staticmethod
    def create_model(activation_name, config):
        """创建指定激活函数的模型"""
        return ActivationMLP(activation_name, config)
    
    @staticmethod
    def create_all_models(config):
        """创建所有激活函数的模型"""
        models = {}
        for activation_name in config.ACTIVATIONS.keys():
            models[activation_name] = ActivationMLP(activation_name, config)
        return models