"""激活函数性能预测模型 - 本地运行版本"""

import numpy as np
import pandas as pd
import json
import pickle
import os
import sys
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 检查并导入必要的机器学习库
try:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, r2_score
    SKLEARN_AVAILABLE = True
except ImportError:
    print("警告: scikit-learn 未安装。将使用理论预测模式。")
    print("请运行: pip install scikit-learn")
    SKLEARN_AVAILABLE = False

class ActivationFunctionPredictor:
    """激活函数性能预测模型"""
    
    def __init__(self):
        # 特征编码器
        self.activation_encoder = LabelEncoder() if SKLEARN_AVAILABLE else None
        self.scaler = StandardScaler() if SKLEARN_AVAILABLE else None
        
        # 预测模型
        self.accuracy_model = None
        self.speed_model = None
        self.stability_model = None
        
        # 训练数据
        self.X = None
        self.y_accuracy = None
        self.y_speed = None
        self.y_stability = None
        
        # 特征列名
        self.feature_names = [
            'activation_type',  # 激活函数类型（编码后）
            'hidden_size_1',    # 第一隐藏层大小
            'hidden_size_2',    # 第二隐藏层大小
            'num_epochs',       # 训练轮数
            'batch_size',       # 批大小
            'learning_rate',    # 学习率
            'weight_decay',     # 权重衰减
            'dataset_size',     # 数据集大小（千为单位）
            'num_classes',      # 类别数
            'input_size'        # 输入维度
        ]
        
        # 激活函数理论特性数据库
        self.activation_properties = self._init_activation_properties()
        
        print("激活函数预测模型初始化完成")
        if not SKLEARN_AVAILABLE:
            print("注意: 使用理论预测模式（scikit-learn 未安装）")
    
    def _init_activation_properties(self):
        """初始化激活函数理论特性数据库"""
        return {
            "ReLU": {
                "range": "(0, +∞)",
                "derivative_range": "(0, 1]",
                "zero_centered": False,
                "gradient_saturation": "partial",
                "computational_cost": 1.0,  # 相对计算成本（1.0为基准）
                "gradient_strength": 0.9,   # 梯度强度（0-1）
                "saturation_risk": 0.3,     # 饱和风险（0-1）
                "dying_neuron_risk": 0.4,   # 死亡神经元风险（0-1）
                "recommended_depth": "deep",  # 推荐网络深度
                "theoretical_accuracy_range": (0.96, 0.985),  # 理论准确率范围
                "theoretical_speed_score": 0.85,  # 理论收敛速度得分
                "theoretical_stability_score": 0.80  # 理论稳定性得分
            },
            "Sigmoid": {
                "range": "(0, 1)",
                "derivative_range": "(0, 0.25]",
                "zero_centered": False,
                "gradient_saturation": "severe",
                "computational_cost": 1.5,
                "gradient_strength": 0.4,
                "saturation_risk": 0.8,
                "dying_neuron_risk": 0.1,
                "recommended_depth": "shallow",
                "theoretical_accuracy_range": (0.88, 0.96),
                "theoretical_speed_score": 0.50,
                "theoretical_stability_score": 0.60
            },
            "Tanh": {
                "range": "(-1, 1)",
                "derivative_range": "(0, 1]",
                "zero_centered": True,
                "gradient_saturation": "moderate",
                "computational_cost": 1.4,
                "gradient_strength": 0.7,
                "saturation_risk": 0.6,
                "dying_neuron_risk": 0.2,
                "recommended_depth": "medium",
                "theoretical_accuracy_range": (0.94, 0.98),
                "theoretical_speed_score": 0.70,
                "theoretical_stability_score": 0.75
            },
            "LeakyReLU": {
                "range": "(-∞, +∞)",
                "derivative_range": "(0.01, 1]",
                "zero_centered": False,
                "gradient_saturation": "none",
                "computational_cost": 1.1,
                "gradient_strength": 0.85,
                "saturation_risk": 0.1,
                "dying_neuron_risk": 0.1,
                "recommended_depth": "deep",
                "theoretical_accuracy_range": (0.965, 0.99),
                "theoretical_speed_score": 0.80,
                "theoretical_stability_score": 0.85
            }
        }
    
    def load_experiment_data(self, results_dir="results"):
        """从实验结果文件加载训练数据"""
        print(f"从 {results_dir} 加载实验数据...")
        
        data_points = []
        result_files = list(Path(results_dir).glob("detailed_result_*.json"))
        result_files += list(Path(results_dir).glob("result_*.json"))
        
        if not result_files:
            print("警告：未找到实验数据文件，将使用理论预测")
            return False
        
        for file in result_files:
            try:
                with open(file, 'r') as f:
                    result = json.load(f)
                
                # 从结果中提取特征
                activation_name = result.get('activation_name')
                history = result.get('history', {})
                
                if not activation_name or not history:
                    continue
                
                # 计算收敛速度（达到最佳准确率90%的epoch）
                val_acc = history.get('val_acc', [])
                if not val_acc:
                    continue
                
                best_acc = max(val_acc)
                target_acc = 0.9 * best_acc
                
                convergence_epoch = None
                for epoch, acc in enumerate(val_acc, 1):
                    if acc >= target_acc:
                        convergence_epoch = epoch
                        break
                convergence_epoch = convergence_epoch or len(val_acc)
                
                # 计算收敛速度得分（epoch越少，得分越高）
                max_epochs = len(val_acc)
                speed_score = max(0, 1 - (convergence_epoch - 1) / max_epochs)
                
                # 计算稳定性（最后5个epoch准确率的标准差，越小越稳定）
                if len(val_acc) >= 5:
                    last_5 = val_acc[-5:]
                    stability = 1 - min(1, np.std(last_5) / 5)  # 标准化到0-1
                else:
                    stability = 0.7  # 默认值
                
                # 获取配置信息
                config = result.get('config', {})
                if isinstance(config, dict):
                    hidden_sizes = config.get('HIDDEN_SIZES', [128, 64])
                    if isinstance(hidden_sizes, list):
                        hidden_size_1 = hidden_sizes[0] if len(hidden_sizes) > 0 else 128
                        hidden_size_2 = hidden_sizes[1] if len(hidden_sizes) > 1 else 64
                    else:
                        hidden_size_1 = 128
                        hidden_size_2 = 64
                else:
                    hidden_size_1 = 128
                    hidden_size_2 = 64
                
                # 创建数据点
                data_point = {
                    'activation_type': activation_name,
                    'hidden_size_1': hidden_size_1,
                    'hidden_size_2': hidden_size_2,
                    'num_epochs': len(val_acc),
                    'batch_size': config.get('BATCH_SIZE', 64) if isinstance(config, dict) else 64,
                    'learning_rate': config.get('LEARNING_RATE', 0.001) if isinstance(config, dict) else 0.001,
                    'weight_decay': config.get('WEIGHT_DECAY', 0.0001) if isinstance(config, dict) else 0.0001,
                    'dataset_size': 70,  # MNIST约70K样本
                    'num_classes': 10,
                    'input_size': 784,
                    'accuracy': best_acc / 100,  # 转换为0-1范围
                    'speed_score': speed_score,
                    'stability_score': stability
                }
                
                data_points.append(data_point)
                
            except Exception as e:
                print(f"加载文件 {file} 失败: {e}")
        
        if not data_points:
            print("警告：未提取到有效数据")
            return False
        
        # 转换为DataFrame
        df = pd.DataFrame(data_points)
        print(f"成功加载 {len(df)} 个数据点")
        
        # 准备特征和标签
        self._prepare_features(df)
        
        return True
    
    def _prepare_features(self, df):
        """准备特征和标签数据"""
        if self.activation_encoder is None:
            print("警告: LabelEncoder 不可用，跳过特征准备")
            return
        
        # 对激活函数类型进行编码
        df['activation_encoded'] = self.activation_encoder.fit_transform(df['activation_type'])
        
        # 特征矩阵
        X = df[self.feature_names[:-3]].copy()  # 排除最后三个特征名（它们是标签）
        X['activation_type'] = df['activation_encoded']
        
        # 添加理论特征
        for idx, row in df.iterrows():
            activation = row['activation_type']
            props = self.activation_properties.get(activation, {})
            
            X.loc[idx, 'gradient_strength'] = props.get('gradient_strength', 0.5)
            X.loc[idx, 'saturation_risk'] = props.get('saturation_risk', 0.5)
            X.loc[idx, 'dying_neuron_risk'] = props.get('dying_neuron_risk', 0.5)
            X.loc[idx, 'computational_cost'] = props.get('computational_cost', 1.0)
            X.loc[idx, 'zero_centered'] = 1 if props.get('zero_centered', False) else 0
        
        self.X = X.values
        self.y_accuracy = df['accuracy'].values
        self.y_speed = df['speed_score'].values
        self.y_stability = df['stability_score'].values
        
        print(f"特征维度: {self.X.shape}")
        print(f"准确率标签: {len(self.y_accuracy)} 个样本")
    
    def train_models(self, test_size=0.2, random_state=42):
        """训练预测模型"""
        if not SKLEARN_AVAILABLE:
            print("错误: scikit-learn 未安装，无法训练模型")
            return False
        
        if self.X is None or len(self.X) < 5:
            print("数据不足，使用理论预测模式")
            return False
        
        print("训练预测模型...")
        
        # 划分训练集和测试集
        X_train, X_test, y_acc_train, y_acc_test, y_speed_train, y_speed_test, y_stab_train, y_stab_test = train_test_split(
            self.X, self.y_accuracy, self.y_speed, self.y_stability,
            test_size=test_size, random_state=random_state
        )
        
        # 标准化特征
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # 训练准确率预测模型
        print("训练准确率预测模型...")
        self.accuracy_model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=random_state
        )
        self.accuracy_model.fit(X_train_scaled, y_acc_train)
        
        # 训练速度预测模型
        print("训练收敛速度预测模型...")
        self.speed_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=5,
            random_state=random_state
        )
        self.speed_model.fit(X_train_scaled, y_speed_train)
        
        # 训练稳定性预测模型
        print("训练稳定性预测模型...")
        self.stability_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=5,
            random_state=random_state
        )
        self.stability_model.fit(X_train_scaled, y_stab_train)
        
        # 评估模型
        self._evaluate_models(X_test_scaled, y_acc_test, y_speed_test, y_stab_test)
        
        return True
    
    def _evaluate_models(self, X_test, y_acc_test, y_speed_test, y_stab_test):
        """评估模型性能"""
        print("\n模型评估结果:")
        print("=" * 50)
        
        # 准确率模型评估
        y_acc_pred = self.accuracy_model.predict(X_test)
        acc_mse = mean_squared_error(y_acc_test, y_acc_pred)
        acc_r2 = r2_score(y_acc_test, y_acc_pred)
        print(f"准确率预测模型:")
        print(f"  MSE: {acc_mse:.6f}")
        print(f"  R²: {acc_r2:.4f}")
        
        # 速度模型评估
        y_speed_pred = self.speed_model.predict(X_test)
        speed_mse = mean_squared_error(y_speed_test, y_speed_pred)
        speed_r2 = r2_score(y_speed_test, y_speed_pred)
        print(f"收敛速度预测模型:")
        print(f"  MSE: {speed_mse:.6f}")
        print(f"  R²: {speed_r2:.4f}")
        
        # 稳定性模型评估
        y_stab_pred = self.stability_model.predict(X_test)
        stab_mse = mean_squared_error(y_stab_test, y_stab_pred)
        stab_r2 = r2_score(y_stab_test, y_stab_pred)
        print(f"稳定性预测模型:")
        print(f"  MSE: {stab_mse:.6f}")
        print(f"  R²: {stab_r2:.4f}")
        
        # 样本预测示例
        print("\n测试集样本预测示例:")
        for i in range(min(3, len(X_test))):
            actual_acc = y_acc_test[i] * 100
            pred_acc = y_acc_pred[i] * 100
            print(f"  样本{i+1}: 实际 {actual_acc:.2f}%, 预测 {pred_acc:.2f}%")
    
    def predict_performance(self, config):
        """
        预测激活函数性能
        
        Args:
            config: 实验配置字典，包含：
                - activation_name: 激活函数名称
                - hidden_sizes: 隐藏层大小列表
                - epochs: 训练轮数
                - batch_size: 批大小
                - learning_rate: 学习率
                - weight_decay: 权重衰减
                - dataset_name: 数据集名称
                - num_classes: 类别数
                - input_size: 输入维度
        """
        activation_name = config.get('activation_name')
        
        if self.accuracy_model is None:
            # 使用理论预测
            print(f"使用理论预测模式")
            return self._theoretical_prediction(activation_name, config)
        
        # 准备特征向量
        feature_vector = self._create_feature_vector(config)
        
        # 标准化特征
        feature_scaled = self.scaler.transform([feature_vector])
        
        # 预测
        accuracy_pred = self.accuracy_model.predict(feature_scaled)[0]
        speed_score = self.speed_model.predict(feature_scaled)[0]
        stability_score = self.stability_model.predict(feature_scaled)[0]
        
        # 转换为可解释的结果
        result = self._format_prediction_results(
            activation_name, accuracy_pred, speed_score, stability_score, config
        )
        
        return result
    
    def _create_feature_vector(self, config):
        """创建特征向量"""
        activation_name = config.get('activation_name', 'ReLU')
        hidden_sizes = config.get('hidden_sizes', [128, 64])
        
        # 获取激活函数理论特性
        props = self.activation_properties.get(activation_name, {})
        
        # 基础特征
        feature_vector = [
            self.activation_encoder.transform([activation_name])[0],  # 激活函数编码
            hidden_sizes[0] if len(hidden_sizes) > 0 else 128,        # 第一隐藏层大小
            hidden_sizes[1] if len(hidden_sizes) > 1 else 64,         # 第二隐藏层大小
            config.get('epochs', 20),                                 # 训练轮数
            config.get('batch_size', 64),                             # 批大小
            config.get('learning_rate', 0.001),                       # 学习率
            config.get('weight_decay', 0.0001),                       # 权重衰减
            self._get_dataset_size(config.get('dataset_name', 'MNIST')),  # 数据集大小
            config.get('num_classes', 10),                            # 类别数
            config.get('input_size', 784),                            # 输入维度
            props.get('gradient_strength', 0.5),                      # 梯度强度
            props.get('saturation_risk', 0.5),                        # 饱和风险
            props.get('dying_neuron_risk', 0.5),                      # 死亡神经元风险
            props.get('computational_cost', 1.0),                     # 计算成本
            1 if props.get('zero_centered', False) else 0             # 是否零中心化
        ]
        
        return np.array(feature_vector)
    
    def _get_dataset_size(self, dataset_name):
        """获取数据集大小（千为单位）"""
        sizes = {
            'MNIST': 70,      # 70K样本
            'CIFAR10': 60,    # 60K样本
            'CIFAR100': 60,   # 60K样本
            'FashionMNIST': 70,
            'KMNIST': 70
        }
        return sizes.get(dataset_name, 50)
    
    def _theoretical_prediction(self, activation_name, config):
        """基于理论特性的预测"""
        props = self.activation_properties.get(activation_name, {})
        
        if not props:
            # 默认预测
            return {
                "activation_name": activation_name,
                "predicted_performance": {
                    "convergence_speed": "中等",
                    "reason": "基于通用理论特性",
                    "final_accuracy": "92-95%",
                    "accuracy_reason": "理论预测范围",
                    "training_stability": "较稳定",
                    "stability_reason": "基于梯度特性分析",
                    "potential_issue": "可能存在梯度问题",
                    "issue_severity": "中等",
                    "mitigation": "建议实验验证"
                }
            }
        
        # 根据配置调整理论预测
        epochs = config.get('epochs', 20)
        hidden_sizes = config.get('hidden_sizes', [128, 64])
        network_depth = len(hidden_sizes)
        
        # 基础理论值
        base_acc_range = props['theoretical_accuracy_range']
        base_speed = props['theoretical_speed_score']
        base_stability = props['theoretical_stability_score']
        
        # 根据网络深度调整
        depth_factor = 1.0
        if network_depth <= 2:
            depth_factor = 1.1  # 浅层网络表现更好
        elif network_depth >= 4:
            depth_factor = 0.9  # 深层网络挑战更大
        
        # 根据训练轮数调整
        epoch_factor = min(1.0, epochs / 20)  # 假设20轮为基准
        
        # 计算调整后的值
        adj_acc_min = base_acc_range[0] * depth_factor * epoch_factor
        adj_acc_max = base_acc_range[1] * min(1.0, depth_factor * 1.05) * min(1.0, epoch_factor * 1.05)
        
        adj_speed = base_speed * depth_factor
        adj_stability = base_stability * depth_factor
        
        # 转换为可解释的结果
        speed_text = self._score_to_speed_text(adj_speed)
        stability_text = self._score_to_stability_text(adj_stability)
        
        # 确定主要问题
        issues = []
        if props['saturation_risk'] > 0.6:
            issues.append("梯度饱和")
        if props['dying_neuron_risk'] > 0.5:
            issues.append("神经元死亡")
        if props['gradient_strength'] < 0.5:
            issues.append("梯度消失")
        
        main_issue = issues[0] if issues else "无明显问题"
        severity = self._risk_to_severity(max(props['saturation_risk'], props['dying_neuron_risk']))
        
        return {
            "activation_name": activation_name,
            "theoretical_properties": {
                "range": props["range"],
                "derivative_range": props["derivative_range"],
                "zero_centered": "是" if props["zero_centered"] else "否",
                "saturation": props["gradient_saturation"],
                "gradient_flow": self._score_to_flow_text(props["gradient_strength"])
            },
            "predicted_performance": {
                "convergence_speed": speed_text,
                "reason": self._get_speed_reason(activation_name, network_depth),
                "final_accuracy": f"{adj_acc_min*100:.1f}-{adj_acc_max*100:.1f}%",
                "accuracy_reason": self._get_accuracy_reason(activation_name, network_depth, epochs),
                "training_stability": stability_text,
                "stability_reason": self._get_stability_reason(activation_name),
                "potential_issue": main_issue,
                "issue_severity": severity,
                "mitigation": self._get_mitigation(activation_name, main_issue)
            },
            "expected_training_curve": self._get_expected_curve(activation_name, speed_text)
        }
    
    def _score_to_speed_text(self, score):
        """将得分转换为速度文本"""
        if score >= 0.8:
            return "快"
        elif score >= 0.6:
            return "中等"
        else:
            return "慢"
    
    def _score_to_stability_text(self, score):
        """将得分转换为稳定性文本"""
        if score >= 0.85:
            return "非常稳定"
        elif score >= 0.7:
            return "稳定"
        elif score >= 0.6:
            return "较稳定"
        else:
            return "不稳定"
    
    def _score_to_flow_text(self, score):
        """将梯度强度得分转换为文本描述"""
        if score >= 0.8:
            return "良好"
        elif score >= 0.6:
            return "中等"
        elif score >= 0.4:
            return "一般"
        else:
            return "差"
    
    def _risk_to_severity(self, risk):
        """将风险值转换为严重程度"""
        if risk >= 0.7:
            return "高"
        elif risk >= 0.4:
            return "中等"
        else:
            return "低"
    
    def _get_speed_reason(self, activation_name, network_depth):
        """获取收敛速度原因"""
        reasons = {
            "ReLU": "梯度不饱和，计算简单",
            "Sigmoid": "梯度饱和问题，但在浅层网络中可能表现尚可",
            "Tanh": "零中心化特性有助于优化",
            "LeakyReLU": "避免死亡神经元，梯度持续"
        }
        
        base_reason = reasons.get(activation_name, "基于激活函数特性")
        
        if network_depth <= 2:
            return f"{base_reason}，浅层网络训练较快"
        elif network_depth >= 4:
            return f"{base_reason}，深层网络可能减慢收敛"
        else:
            return base_reason
    
    def _get_accuracy_reason(self, activation_name, network_depth, epochs):
        """获取准确率预测原因"""
        reasons = {
            "ReLU": "缓解梯度消失，适合深层网络",
            "Sigmoid": "在浅层网络中仍能达到不错效果",
            "Tanh": "零中心化输出有助于训练",
            "LeakyReLU": "结合ReLU优点，避免其缺点"
        }
        
        base_reason = reasons.get(activation_name, "基于激活函数特性")
        
        depth_note = ""
        if network_depth <= 2:
            depth_note = "浅层网络通常表现良好"
        elif network_depth >= 4:
            depth_note = "深层网络可能面临挑战"
        
        epoch_note = f"训练{epochs}轮"
        
        return f"{base_reason}。{depth_note} {epoch_note}。"
    
    def _get_stability_reason(self, activation_name):
        """获取稳定性原因"""
        reasons = {
            "ReLU": "梯度一致，但可能受死亡神经元影响",
            "Sigmoid": "梯度变化大，训练可能不稳定",
            "Tanh": "对称梯度分布，相对稳定",
            "LeakyReLU": "避免ReLU问题，训练稳定"
        }
        return reasons.get(activation_name, "基于梯度特性分析")
    
    def _get_mitigation(self, activation_name, issue):
        """获取问题缓解建议"""
        mitigations = {
            "梯度饱和": "使用批归一化或残差连接",
            "神经元死亡": "使用LeakyReLU或适当的初始化",
            "梯度消失": "使用ReLU家族或合适的权重初始化"
        }
        
        specific_mitigations = {
            "ReLU": "使用He初始化，考虑LeakyReLU变体",
            "Sigmoid": "限制网络深度，使用批归一化",
            "Tanh": "配合批归一化使用",
            "LeakyReLU": "保持α=0.01通常效果良好"
        }
        
        general = mitigations.get(issue, "调整学习率或网络结构")
        specific = specific_mitigations.get(activation_name, "")
        
        return f"{general}。{specific}"
    
    def _get_expected_curve(self, activation_name, speed_text):
        """获取预期训练曲线描述"""
        curves = {
            "ReLU": {
                "fast": {
                    "early_stage": "快速下降",
                    "mid_stage": "稳定提升",
                    "late_stage": "缓慢收敛"
                },
                "medium": {
                    "early_stage": "稳定下降",
                    "mid_stage": "持续提升",
                    "late_stage": "逐渐收敛"
                }
            },
            "Sigmoid": {
                "fast": {
                    "early_stage": "较快下降",
                    "mid_stage": "稳定提升",
                    "late_stage": "可能停滞"
                },
                "medium": {
                    "early_stage": "缓慢下降",
                    "mid_stage": "逐渐提升",
                    "late_stage": "缓慢收敛"
                },
                "slow": {
                    "early_stage": "缓慢下降",
                    "mid_stage": "波动提升",
                    "late_stage": "可能不收敛"
                }
            }
        }
        
        # 默认曲线
        default_curve = {
            "early_stage": "稳定下降",
            "mid_stage": "持续提升",
            "late_stage": "逐渐收敛"
        }
        
        activation_curves = curves.get(activation_name, {})
        return activation_curves.get(speed_text, default_curve)
    
    def _format_prediction_results(self, activation_name, accuracy, speed_score, stability_score, config):
        """格式化预测结果"""
        # 转换为百分比
        accuracy_pct = accuracy * 100
        
        # 确定准确率范围（基于预测值和置信度）
        accuracy_range = self._calculate_accuracy_range(accuracy_pct, activation_name)
        
        # 转换为文本
        speed_text = self._score_to_speed_text(speed_score)
        stability_text = self._score_to_stability_text(stability_score)
        
        # 获取理论特性
        props = self.activation_properties.get(activation_name, {})
        
        return {
            "activation_name": activation_name,
            "theoretical_properties": {
                "range": props.get("range", "(0, +∞)"),
                "derivative_range": props.get("derivative_range", "(0, 1]"),
                "zero_centered": "是" if props.get("zero_centered", False) else "否",
                "saturation": props.get("gradient_saturation", "部分饱和"),
                "gradient_flow": self._score_to_flow_text(props.get("gradient_strength", 0.5))
            },
            "predicted_performance": {
                "convergence_speed": speed_text,
                "reason": f"基于机器学习模型预测，得分: {speed_score:.3f}",
                "final_accuracy": accuracy_range,
                "accuracy_reason": f"模型预测值: {accuracy_pct:.2f}%，考虑实验条件调整",
                "training_stability": stability_text,
                "stability_reason": f"稳定性得分: {stability_score:.3f}",
                "potential_issue": self._identify_potential_issue(activation_name, speed_score, stability_score),
                "issue_severity": self._assess_issue_severity(activation_name),
                "mitigation": "根据模型预测调整超参数"
            },
            "model_confidence": {
                "accuracy_confidence": 0.95,
                "speed_confidence": 0.85,
                "stability_confidence": 0.80
            }
        }
    
    def _calculate_accuracy_range(self, predicted_accuracy, activation_name):
        """计算准确率预测范围"""
        # 基于激活函数类型和预测值确定范围
        props = self.activation_properties.get(activation_name, {})
        base_range = props.get('theoretical_accuracy_range', (0.9, 0.98))
        
        # 计算范围
        lower_bound = max(base_range[0] * 100, predicted_accuracy - 1.5)
        upper_bound = min(base_range[1] * 100, predicted_accuracy + 1.5)
        
        # 确保范围合理
        lower_bound = max(80, lower_bound)
        upper_bound = min(100, upper_bound)
        
        return f"{lower_bound:.1f}-{upper_bound:.1f}%"
    
    def _identify_potential_issue(self, activation_name, speed_score, stability_score):
        """识别潜在问题"""
        props = self.activation_properties.get(activation_name, {})
        
        issues = []
        
        if speed_score < 0.6:
            issues.append("收敛速度可能较慢")
        
        if stability_score < 0.7:
            issues.append("训练可能不稳定")
        
        if props.get('saturation_risk', 0.5) > 0.6:
            issues.append("梯度饱和风险")
        
        if props.get('dying_neuron_risk', 0.5) > 0.5:
            issues.append("神经元死亡风险")
        
        return issues[0] if issues else "无明显问题"
    
    def _assess_issue_severity(self, activation_name):
        """评估问题严重程度"""
        props = self.activation_properties.get(activation_name, {})
        
        max_risk = max(
            props.get('saturation_risk', 0.5),
            props.get('dying_neuron_risk', 0.5)
        )
        
        return self._risk_to_severity(max_risk)
    
    def save_model(self, filename=None):
        """保存训练好的模型"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"activation_predictor_model_{timestamp}.pkl"
        
        model_data = {
            'accuracy_model': self.accuracy_model,
            'speed_model': self.speed_model,
            'stability_model': self.stability_model,
            'activation_encoder': self.activation_encoder,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'activation_properties': self.activation_properties,
            'saved_at': datetime.now().isoformat()
        }
        
        Path("models").mkdir(exist_ok=True)
        
        with open(f"models/{filename}", 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"模型已保存到: models/{filename}")
        return filename
    
    def load_model(self, filename):
        """加载已训练的模型"""
        try:
            with open(f"models/{filename}", 'rb') as f:
                model_data = pickle.load(f)
            
            self.accuracy_model = model_data['accuracy_model']
            self.speed_model = model_data['speed_model']
            self.stability_model = model_data['stability_model']
            self.activation_encoder = model_data['activation_encoder']
            self.scaler = model_data['scaler']
            self.feature_names = model_data['feature_names']
            self.activation_properties = model_data.get('activation_properties', self.activation_properties)
            
            print(f"模型已加载: {filename}")
            return True
            
        except Exception as e:
            print(f"加载模型失败: {e}")
            return False
    
    def predict_all_activations(self, config_template):
        """预测所有激活函数的性能"""
        activations = ["ReLU", "Sigmoid", "Tanh", "LeakyReLU"]
        predictions = {}
        
        print("开始预测各激活函数性能...")
        print("=" * 60)
        
        for act_name in activations:
            config = config_template.copy()
            config['activation_name'] = act_name
            
            prediction = self.predict_performance(config)
            predictions[act_name] = prediction
            
            # 打印简要信息
            perf = prediction['predicted_performance']
            print(f"{act_name}:")
            print(f"  收敛速度: {perf['convergence_speed']}")
            print(f"  最终准确率: {perf['final_accuracy']}")
            print(f"  主要问题: {perf['potential_issue']}")
            print("-" * 40)
        
        return predictions


# 与原始接口兼容的包装器
class LocalLLMPredictor:
    """本地预测器 - 保持与原始LLMPredictor相同的接口"""
    
    def __init__(self, use_ml_model=True):
        self.use_ml_model = use_ml_model
        self.predictor = ActivationFunctionPredictor()
        
        # 尝试加载现有模型或训练新模型
        if use_ml_model and SKLEARN_AVAILABLE:
            self._initialize_ml_model()
        else:
            print("使用理论预测模式")
    
    def _initialize_ml_model(self):
        """初始化机器学习模型"""
        # 首先尝试加载现有模型
        Path("models").mkdir(exist_ok=True)
        model_files = list(Path("models").glob("activation_predictor_model_*.pkl"))
        
        if model_files:
            # 加载最新的模型
            latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
            if self.predictor.load_model(latest_model.name):
                print(f"✓ 已加载现有预测模型: {latest_model.name}")
                return
        
        # 如果没有现有模型，尝试训练新模型
        print("尝试训练新的预测模型...")
        has_data = self.predictor.load_experiment_data()
        
        if has_data and self.predictor.X is not None and len(self.predictor.X) >= 10:  # 至少有10个样本
            self.predictor.train_models()
            self.predictor.save_model()
        else:
            print("数据不足，将使用理论预测模式")
            self.use_ml_model = False
    
    @staticmethod
    def get_predictions(config=None):
        """获取各激活函数的预测表现（保持与原始接口兼容）"""
        predictor = LocalLLMPredictor()
        
        # 默认配置
        if config is None:
            config = {
                'hidden_sizes': [128, 64],
                'epochs': 20,
                'batch_size': 64,
                'learning_rate': 0.001,
                'weight_decay': 0.0001,
                'dataset_name': 'MNIST',
                'num_classes': 10,
                'input_size': 784
            }
        
        # 预测所有激活函数
        predictions = predictor.predict_all_activations(config)
        
        # 转换为原始格式
        return LocalLLMPredictor._convert_to_original_format(predictions)
    
    def predict_all_activations(self, config_template):
        """预测所有激活函数的性能"""
        predictions = {}
        
        for act_name in ["ReLU", "Sigmoid", "Tanh", "LeakyReLU"]:
            config = config_template.copy()
            config['activation_name'] = act_name
            
            prediction = self.predictor.predict_performance(config)
            predictions[act_name] = prediction
        
        return predictions
    
    @staticmethod
    def _convert_to_original_format(predictions):
        """转换为原始预测格式"""
        original_format = {}
        
        for act_name, prediction in predictions.items():
            # 确保有 expected_training_curve
            if 'expected_training_curve' not in prediction:
                prediction['expected_training_curve'] = {
                    "early_stage": "稳定下降",
                    "mid_stage": "持续提升", 
                    "late_stage": "逐渐收敛"
                }
            
            original_format[act_name] = {
                "theoretical_properties": prediction.get("theoretical_properties", {}),
                "predicted_performance": prediction.get("predicted_performance", {}),
                "expected_training_curve": prediction.get("expected_training_curve", {})
            }
        
        return original_format
    
    @staticmethod
    def save_predictions(predictions, filename=None):
        """保存预测结果到JSON文件"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"local_predictions_{timestamp}.json"
        
        data = {
            "timestamp": datetime.now().isoformat(),
            "predictions": predictions,
            "disclaimer": "这些预测基于本地机器学习模型和理论特性，实际表现可能因具体实验条件而异"
        }
        
        Path("results").mkdir(exist_ok=True)
        
        with open(f"results/{filename}", 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"预测已保存到: results/{filename}")
        return filename
    
    @staticmethod
    def load_predictions(filename):
        """从JSON文件加载预测结果"""
        with open(f"results/{filename}", 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data["predictions"]


# 测试函数
def test_local_predictor():
    """测试本地预测器"""
    print("测试本地激活函数预测模型")
    print("=" * 60)
    
    # 创建预测器
    predictor = LocalLLMPredictor()
    
    # 配置示例
    config = {
        'hidden_sizes': [128, 64],
        'epochs': 20,
        'batch_size': 64,
        'learning_rate': 0.001,
        'weight_decay': 0.0001,
        'dataset_name': 'MNIST',
        'num_classes': 10,
        'input_size': 784
    }
    
    # 获取预测
    predictions = predictor.predict_all_activations(config)
    
    # 转换为原始格式并保存
    original_predictions = LocalLLMPredictor._convert_to_original_format(predictions)
    filename = LocalLLMPredictor.save_predictions(original_predictions)
    
    # 打印摘要
    print("\n预测摘要:")
    print("=" * 60)
    for act_name, pred in original_predictions.items():
        perf = pred["predicted_performance"]
        print(f"{act_name}:")
        print(f"  收敛速度: {perf['convergence_speed']}")
        print(f"  最终准确率: {perf['final_accuracy']}")
        print(f"  主要问题: {perf['potential_issue']}")
        print("-" * 40)
    
    return original_predictions


def main():
    """主函数"""
    print("激活函数本地预测系统")
    print("=" * 60)
    
    # 检查scikit-learn是否可用
    if not SKLEARN_AVAILABLE:
        print("警告: scikit-learn 未安装，只能使用理论预测模式。")
        print("要使用机器学习预测，请运行: pip install scikit-learn")
        print()
    
    print("\n选项:")
    print("1. 使用默认配置进行预测")
    print("2. 自定义配置进行预测")
    print("3. 训练新的预测模型")
    print("4. 退出")
    
    choice = input("\n请选择 (1-4): ").strip()
    
    if choice == "1":
        predictions = test_local_predictor()
        
        # 询问是否查看详细结果
        view_details = input("\n是否查看详细预测结果? (y/n): ").strip().lower()
        if view_details == 'y':
            for act_name, pred in predictions.items():
                print(f"\n{act_name} 详细预测:")
                print("-" * 40)
                for key, value in pred.items():
                    if isinstance(value, dict):
                        print(f"{key}:")
                        for k, v in value.items():
                            print(f"  {k}: {v}")
                    else:
                        print(f"{key}: {value}")
        
        print(f"\n预测结果已保存到 results/ 目录")
    
    elif choice == "2":
        print("\n请输入实验配置:")
        
        config = {}
        hidden_sizes_input = input("隐藏层大小 (如 '128 64'): ").strip()
        if hidden_sizes_input:
            config['hidden_sizes'] = list(map(int, hidden_sizes_input.split()))
        else:
            config['hidden_sizes'] = [128, 64]
        
        config['epochs'] = int(input("训练轮数 (默认20): ") or "20")
        config['batch_size'] = int(input("批大小 (默认64): ") or "64")
        config['learning_rate'] = float(input("学习率 (默认0.001): ") or "0.001")
        config['weight_decay'] = float(input("权重衰减 (默认0.0001): ") or "0.0001")
        config['dataset_name'] = input("数据集名称 (默认MNIST): ") or "MNIST"
        config['num_classes'] = int(input("类别数 (默认10): ") or "10")
        
        # 根据数据集设置输入大小
        if config['dataset_name'] == 'MNIST':
            config['input_size'] = 784
        elif config['dataset_name'] == 'CIFAR10':
            config['input_size'] = 3072
        else:
            config['input_size'] = int(input("输入维度 (默认784): ") or "784")
        
        # 获取预测
        predictor = LocalLLMPredictor()
        predictions = predictor.predict_all_activations(config)
        original_predictions = LocalLLMPredictor._convert_to_original_format(predictions)
        
        # 打印结果
        print("\n预测结果:")
        print("=" * 60)
        for act_name, pred in original_predictions.items():
            perf = pred["predicted_performance"]
            print(f"{act_name}:")
            print(f"  收敛速度: {perf['convergence_speed']}")
            print(f"  最终准确率: {perf['final_accuracy']}")
            print(f"  稳定性: {perf['training_stability']}")
            print(f"  主要问题: {perf['potential_issue']}")
            print("-" * 40)
        
        # 保存结果
        save = input("\n是否保存预测结果? (y/n): ").strip().lower()
        if save == 'y':
            filename = LocalLLMPredictor.save_predictions(original_predictions)
            print(f"预测已保存到: {filename}")
    
    elif choice == "3":
        if not SKLEARN_AVAILABLE:
            print("错误: scikit-learn 未安装，无法训练模型")
            print("请运行: pip install scikit-learn")
            return
        
        print("\n训练新的预测模型...")
        
        # 确保目录存在
        Path("models").mkdir(exist_ok=True)
        Path("results").mkdir(exist_ok=True)
        
        # 创建预测器并训练
        predictor = ActivationFunctionPredictor()
        has_data = predictor.load_experiment_data()
        
        if has_data and predictor.X is not None:
            if len(predictor.X) >= 5:
                predictor.train_models()
                filename = predictor.save_model()
                print(f"新模型已训练并保存: {filename}")
                
                # 测试新模型
                print("\n测试新模型预测:")
                test_config = {
                    'activation_name': 'ReLU',
                    'hidden_sizes': [128, 64],
                    'epochs': 20,
                    'batch_size': 64,
                    'learning_rate': 0.001,
                    'weight_decay': 0.0001,
                    'dataset_name': 'MNIST',
                    'num_classes': 10,
                    'input_size': 784
                }
                
                prediction = predictor.predict_performance(test_config)
                perf = prediction['predicted_performance']
                print(f"ReLU 预测结果:")
                print(f"  收敛速度: {perf['convergence_speed']}")
                print(f"  最终准确率: {perf['final_accuracy']}")
            else:
                print(f"数据不足，只有 {len(predictor.X) if predictor.X is not None else 0} 个样本")
                print("建议先运行更多实验收集数据")
        else:
            print("没有找到足够的实验数据")
            print("请先运行实验生成 detailed_result_*.json 文件")
            print("至少需要5个不同的实验数据点")
    
    elif choice == "4":
        print("退出程序")
    
    else:
        print("无效选择")


if __name__ == "__main__":
    # 确保必要的目录存在
    for dir_name in ["results", "models", "data"]:
        Path(dir_name).mkdir(exist_ok=True)
    
    main()