"""深入分析模块"""

import numpy as np
import pandas as pd
from scipy import stats
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any

class ActivationFunctionAnalyzer:
    """激活函数深入分析器"""
    
    def __init__(self, config):
        self.config = config
    
    def analyze_theoretical_properties(self):
        """分析激活函数的理论特性"""
        properties = {}
        
        # ReLU
        properties['ReLU'] = {
            'function': 'f(x) = max(0, x)',
            'derivative': "f'(x) = 1 if x > 0 else 0",
            'range': '(0, +∞)',
            'zero_centered': 'No',
            'gradient_saturation': 'Partial (negative side)',
            'computational_cost': 'Low',
            'common_issues': 'Dying ReLU, not zero-centered',
            'recommended_usage': 'Hidden layers, deep networks',
            'initialization_importance': 'High (He initialization)'
        }
        
        # Sigmoid
        properties['Sigmoid'] = {
            'function': 'f(x) = 1 / (1 + exp(-x))',
            'derivative': "f'(x) = f(x)(1 - f(x))",
            'range': '(0, 1)',
            'zero_centered': 'No',
            'gradient_saturation': 'Severe (both sides)',
            'computational_cost': 'Medium (exp calculation)',
            'common_issues': 'Vanishing gradient, slow convergence',
            'recommended_usage': 'Output layer for binary classification',
            'initialization_importance': 'Medium'
        }
        
        # Tanh
        properties['Tanh'] = {
            'function': 'f(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))',
            'derivative': "f'(x) = 1 - f(x)²",
            'range': '(-1, 1)',
            'zero_centered': 'Yes',
            'gradient_saturation': 'Moderate (both sides)',
            'computational_cost': 'Medium (exp calculation)',
            'common_issues': 'Vanishing gradient (less severe than sigmoid)',
            'recommended_usage': 'Hidden layers, RNNs',
            'initialization_importance': 'Medium'
        }
        
        # LeakyReLU
        properties['LeakyReLU'] = {
            'function': 'f(x) = x if x > 0 else αx',
            'derivative': "f'(x) = 1 if x > 0 else α",
            'range': '(-∞, +∞)',
            'zero_centered': 'No',
            'gradient_saturation': 'None',
            'computational_cost': 'Low',
            'common_issues': 'Parameter α needs tuning',
            'recommended_usage': 'Hidden layers, when ReLU fails',
            'initialization_importance': 'Medium'
        }
        
        return properties
    
    def analyze_gradient_flow(self, results_dict: Dict) -> Dict:
        """分析梯度流动特性"""
        gradient_analysis = {}
        
        for act_name, result in results_dict.items():
            if 'gradient_stats' in result and result['gradient_stats']:
                grad_stats = result['gradient_stats']
                
                # 收集所有梯度值
                all_gradients = []
                all_gradient_means = []
                
                for stat in grad_stats:
                    if 'grad_info' in stat:
                        for param_name, grad_info in stat['grad_info'].items():
                            if 'mean' in grad_info:
                                grad_value = grad_info['mean']
                                all_gradients.append(abs(grad_value))
                                all_gradient_means.append(grad_value)
                
                if all_gradients:
                    gradient_analysis[act_name] = {
                        'mean_gradient': np.mean(all_gradients),
                        'std_gradient': np.std(all_gradients),
                        'gradient_range': (np.min(all_gradients), np.max(all_gradients)),
                        'gradient_mean_raw': np.mean(all_gradient_means),
                        'gradient_std_raw': np.std(all_gradient_means),
                        'gradient_skewness': stats.skew(all_gradients),
                        'gradient_kurtosis': stats.kurtosis(all_gradients),
                        'zero_gradient_ratio': sum(1 for g in all_gradient_means if abs(g) < 1e-8) / len(all_gradient_means),
                        'vanishing_gradient_risk': 'High' if np.mean(all_gradients) < 1e-4 else 'Medium' if np.mean(all_gradients) < 1e-3 else 'Low',
                        'exploding_gradient_risk': 'High' if np.max(all_gradients) > 1.0 else 'Medium' if np.max(all_gradients) > 0.1 else 'Low'
                    }
        
        return gradient_analysis
    
    def analyze_convergence_behavior(self, results_dict: Dict) -> Dict:
        """分析收敛行为"""
        convergence_analysis = {}
        
        for act_name, result in results_dict.items():
            history = result['history']
            val_acc = history['val_acc']
            train_loss = history['train_loss']
            val_loss = history['val_loss']
            
            # 收敛速度指标
            best_acc = max(val_acc)
            target_acc_90 = 0.9 * best_acc
            target_acc_95 = 0.95 * best_acc
            
            epoch_to_90 = None
            epoch_to_95 = None
            
            for epoch, acc in enumerate(val_acc, 1):
                if epoch_to_90 is None and acc >= target_acc_90:
                    epoch_to_90 = epoch
                if epoch_to_95 is None and acc >= target_acc_95:
                    epoch_to_95 = epoch
            
            # 收敛稳定性指标
            if len(val_acc) >= 10:
                last_10_acc = val_acc[-10:]
                last_10_loss = val_loss[-10:]
                
                acc_std = np.std(last_10_acc)
                loss_std = np.std(last_10_loss)
                
                # 计算震荡程度
                acc_oscillation = np.mean(np.abs(np.diff(last_10_acc)))
                loss_oscillation = np.mean(np.abs(np.diff(last_10_loss)))
            else:
                acc_std = np.std(val_acc)
                loss_std = np.std(val_loss)
                acc_oscillation = np.mean(np.abs(np.diff(val_acc))) if len(val_acc) > 1 else 0
                loss_oscillation = np.mean(np.abs(np.diff(val_loss))) if len(val_loss) > 1 else 0
            
            # 过拟合分析
            final_train_acc = result['final_train_acc']
            final_val_acc = result['final_val_acc']
            overfitting_gap = final_train_acc - final_val_acc
            
            convergence_analysis[act_name] = {
                'best_accuracy': best_acc,
                'epochs_to_90_percent': epoch_to_90,
                'epochs_to_95_percent': epoch_to_95,
                'convergence_speed': 'Fast' if epoch_to_90 <= 5 else 'Medium' if epoch_to_90 <= 10 else 'Slow',
                'accuracy_stability': acc_std,
                'loss_stability': loss_std,
                'accuracy_oscillation': acc_oscillation,
                'loss_oscillation': loss_oscillation,
                'overfitting_gap': overfitting_gap,
                'overfitting_level': 'High' if overfitting_gap > 5 else 'Medium' if overfitting_gap > 2 else 'Low',
                'final_generalization_gap': overfitting_gap,
                'training_efficiency': result['total_training_time'] / (epoch_to_90 if epoch_to_90 else len(val_acc))
            }
        
        return convergence_analysis
    
    def analyze_sensitivity_to_hyperparameters(self, results_dict: Dict) -> Dict:
        """分析对超参数的敏感性"""
        sensitivity_analysis = {}
        
        # 基于训练历史分析敏感性
        for act_name, result in results_dict.items():
            history = result['history']
            
            # 分析学习率变化的影响
            lr_history = history['learning_rate']
            lr_changes = np.abs(np.diff(lr_history))
            lr_sensitivity = np.mean(lr_changes) if len(lr_changes) > 0 else 0
            
            # 分析损失对学习率的敏感性
            if len(history['val_loss']) >= 2:
                loss_changes = np.diff(history['val_loss'])
                lr_loss_correlation = np.corrcoef(lr_history[:len(loss_changes)], loss_changes)[0, 1] if len(lr_history) == len(loss_changes) + 1 else 0
            else:
                lr_loss_correlation = 0
            
            # 训练稳定性指标
            val_acc = history['val_acc']
            if len(val_acc) >= 5:
                early_performance = np.mean(val_acc[:5])
                late_performance = np.mean(val_acc[-5:])
                improvement_ratio = (late_performance - early_performance) / early_performance
            else:
                improvement_ratio = 0
            
            sensitivity_analysis[act_name] = {
                'lr_sensitivity': lr_sensitivity,
                'lr_loss_correlation': lr_loss_correlation,
                'improvement_ratio': improvement_ratio,
                'requires_careful_tuning': 'Yes' if lr_sensitivity > 0.1 else 'No',
                'robust_to_lr_changes': 'Yes' if abs(lr_loss_correlation) < 0.3 else 'No',
                'consistent_improvement': 'Yes' if improvement_ratio > 0.1 else 'No'
            }
        
        return sensitivity_analysis
    
    def analyze_layer_wise_behavior(self, results_dict: Dict) -> Dict:
        """分析不同层的梯度行为（如果数据可用）"""
        layer_analysis = {}
        
        for act_name, result in results_dict.items():
            if 'gradient_stats' in result and result['gradient_stats']:
                grad_stats = result['gradient_stats']
                
                # 按参数名分组（假设参数名包含层信息）
                layer_gradients = {}
                
                for stat in grad_stats[:10]:  # 分析前10个检查点
                    if 'grad_info' in stat:
                        for param_name, grad_info in stat['grad_info'].items():
                            # 提取层信息
                            if 'weight' in param_name:
                                # 假设参数名格式: layers.0.weight, layers.1.weight, 等
                                parts = param_name.split('.')
                                if len(parts) >= 2 and parts[0] == 'layers':
                                    layer_idx = parts[1]
                                    if layer_idx not in layer_gradients:
                                        layer_gradients[layer_idx] = []
                                    
                                    if 'mean' in grad_info:
                                        layer_gradients[layer_idx].append(abs(grad_info['mean']))
                
                # 计算每层的统计信息
                layer_stats = {}
                for layer_idx, gradients in layer_gradients.items():
                    if gradients:
                        layer_stats[layer_idx] = {
                            'mean_gradient': np.mean(gradients),
                            'std_gradient': np.std(gradients),
                            'gradient_decay_ratio': gradients[0] / gradients[-1] if gradients[-1] != 0 else float('inf')
                        }
                
                layer_analysis[act_name] = layer_stats
        
        return layer_analysis
    
    def perform_statistical_tests(self, results_dict: Dict) -> Dict:
        """执行统计检验"""
        statistical_tests = {}
        
        # 准备准确率数据
        accuracy_data = {}
        for act_name, result in results_dict.items():
            accuracy_data[act_name] = result['history']['val_acc']
        
        # 对每组激活函数进行t检验
        act_names = list(accuracy_data.keys())
        
        for i in range(len(act_names)):
            for j in range(i + 1, len(act_names)):
                act1 = act_names[i]
                act2 = act_names[j]
                
                data1 = accuracy_data[act1]
                data2 = accuracy_data[act2]
                
                # 确保数据长度相同
                min_len = min(len(data1), len(data2))
                data1_trunc = data1[:min_len]
                data2_trunc = data2[:min_len]
                
                # t检验
                t_stat, p_value = stats.ttest_ind(data1_trunc, data2_trunc)
                
                # Mann-Whitney U检验（非参数）
                u_stat, u_p_value = stats.mannwhitneyu(data1_trunc, data2_trunc)
                
                key = f"{act1}_vs_{act2}"
                statistical_tests[key] = {
                    't_test': {
                        't_statistic': t_stat,
                        'p_value': p_value,
                        'significant': p_value < 0.05,
                        'effect_size': abs(t_stat) / np.sqrt(len(data1_trunc) + len(data2_trunc) - 2)
                    },
                    'mann_whitney': {
                        'u_statistic': u_stat,
                        'p_value': u_p_value,
                        'significant': u_p_value < 0.05
                    },
                    'mean_difference': np.mean(data1_trunc) - np.mean(data2_trunc),
                    'relative_improvement': (np.mean(data1_trunc) - np.mean(data2_trunc)) / np.mean(data2_trunc) * 100
                }
        
        return statistical_tests
    
    def generate_comprehensive_analysis(self, results_dict: Dict, llm_predictions: Dict) -> Dict:
        """生成综合分析报告"""
        analysis = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'dataset': self.config.DATASET_NAME,
            'model_architecture': f"MLP with hidden sizes {self.config.HIDDEN_SIZES}",
            'theoretical_properties': self.analyze_theoretical_properties(),
            'gradient_flow_analysis': self.analyze_gradient_flow(results_dict),
            'convergence_analysis': self.analyze_convergence_behavior(results_dict),
            'sensitivity_analysis': self.analyze_sensitivity_to_hyperparameters(results_dict),
            'layer_wise_analysis': self.analyze_layer_wise_behavior(results_dict),
            'statistical_tests': self.perform_statistical_tests(results_dict),
            'llm_prediction_accuracy': self.evaluate_llm_predictions(results_dict, llm_predictions),
            'recommendations': self.generate_recommendations(results_dict),
            'limitations': self.identify_limitations(),
            'future_work': self.suggest_future_work()
        }
        
        return analysis
    
    def evaluate_llm_predictions(self, results_dict: Dict, llm_predictions: Dict) -> Dict:
        """评估大模型预测的准确性"""
        evaluation = {
            'correct_predictions': [],
            'incorrect_predictions': [],
            'partial_correct_predictions': [],
            'accuracy_metrics': {},
            'detailed_comparison': {}
        }
        
        convergence_analysis = self.analyze_convergence_behavior(results_dict)
        
        for act_name in results_dict.keys():
            if act_name in llm_predictions:
                exp_result = convergence_analysis[act_name]
                llm_pred = llm_predictions[act_name]['predicted_performance']
                
                # 收敛速度对比
                pred_speed = llm_pred['convergence_speed']
                actual_speed = exp_result['convergence_speed']
                speed_match = pred_speed == actual_speed
                
                # 准确率对比
                pred_acc_str = llm_pred['final_accuracy']
                actual_acc = exp_result['best_accuracy']
                
                # 从预测字符串提取数值范围
                import re
                numbers = re.findall(r'\d+\.?\d*', pred_acc_str)
                if numbers:
                    if len(numbers) == 1:
                        pred_range = (float(numbers[0]) - 2, float(numbers[0]) + 2)
                    else:
                        pred_range = (float(numbers[0]), float(numbers[1]))
                    
                    acc_in_range = pred_range[0] <= actual_acc <= pred_range[1]
                else:
                    acc_in_range = False
                
                # 问题预测对比
                predicted_issue = llm_pred['potential_issue']
                actual_gradient_risk = 'N/A'
                
                if 'gradient_flow_analysis' in self.analyze_gradient_flow(results_dict):
                    gradient_analysis = self.analyze_gradient_flow(results_dict)
                    if act_name in gradient_analysis:
                        grad_risk = gradient_analysis[act_name]['vanishing_gradient_risk']
                        exploding_risk = gradient_analysis[act_name]['exploding_gradient_risk']
                        
                        if 'vanishing' in predicted_issue.lower() and grad_risk == 'High':
                            issue_match = True
                        elif 'exploding' in predicted_issue.lower() and exploding_risk == 'High':
                            issue_match = True
                        else:
                            issue_match = False
                    else:
                        issue_match = False
                else:
                    issue_match = 'Unknown'
                
                # 总体匹配度
                overall_match = speed_match and acc_in_range
                
                evaluation['detailed_comparison'][act_name] = {
                    'convergence_speed': {
                        'predicted': pred_speed,
                        'actual': actual_speed,
                        'match': speed_match
                    },
                    'accuracy': {
                        'predicted_range': pred_acc_str,
                        'actual': f"{actual_acc:.2f}%",
                        'in_range': acc_in_range
                    },
                    'issues': {
                        'predicted': predicted_issue,
                        'actual_gradient_risk': actual_gradient_risk,
                        'match': issue_match
                    },
                    'overall_match': overall_match
                }
                
                if overall_match:
                    evaluation['correct_predictions'].append(act_name)
                elif speed_match or acc_in_range:
                    evaluation['partial_correct_predictions'].append(act_name)
                else:
                    evaluation['incorrect_predictions'].append(act_name)
        
        # 计算准确率指标
        total = len(evaluation['detailed_comparison'])
        if total > 0:
            evaluation['accuracy_metrics'] = {
                'total_predictions': total,
                'correct_predictions': len(evaluation['correct_predictions']),
                'partial_correct': len(evaluation['partial_correct_predictions']),
                'incorrect_predictions': len(evaluation['incorrect_predictions']),
                'overall_accuracy': len(evaluation['correct_predictions']) / total * 100,
                'partial_accuracy': (len(evaluation['correct_predictions']) + 
                                   len(evaluation['partial_correct_predictions'])) / total * 100
            }
        
        return evaluation
    
    def generate_recommendations(self, results_dict: Dict) -> Dict:
        """基于分析结果生成建议"""
        convergence_analysis = self.analyze_convergence_behavior(results_dict)
        sensitivity_analysis = self.analyze_sensitivity_to_hyperparameters(results_dict)
        
        # 按性能排序
        sorted_acts = sorted(convergence_analysis.items(), 
                           key=lambda x: x[1]['best_accuracy'], 
                           reverse=True)
        
        recommendations = {
            'best_overall': sorted_acts[0][0] if sorted_acts else 'N/A',
            'fastest_convergence': min(convergence_analysis.items(), 
                                      key=lambda x: x[1]['epochs_to_90_percent'] or float('inf'))[0] 
                                   if convergence_analysis else 'N/A',
            'most_stable': min(convergence_analysis.items(), 
                             key=lambda x: x[1]['accuracy_stability'])[0] 
                          if convergence_analysis else 'N/A',
            'least_overfitting': min(convergence_analysis.items(), 
                                   key=lambda x: x[1]['overfitting_gap'])[0] 
                                if convergence_analysis else 'N/A',
            'easiest_to_tune': min(sensitivity_analysis.items(), 
                                 key=lambda x: x[1]['lr_sensitivity'])[0] 
                              if sensitivity_analysis else 'N/A',
            'specific_recommendations': {}
        }
        
        # 为每个激活函数生成具体建议
        for act_name in results_dict.keys():
            conv_info = convergence_analysis.get(act_name, {})
            sens_info = sensitivity_analysis.get(act_name, {})
            
            specific_rec = []
            
            if conv_info.get('overfitting_level') == 'High':
                specific_rec.append("使用更强的正则化（如dropout、权重衰减）")
            
            if conv_info.get('convergence_speed') == 'Slow':
                specific_rec.append("尝试更高的初始学习率或使用学习率预热")
            
            if sens_info.get('requires_careful_tuning') == 'Yes':
                specific_rec.append("需要仔细调整学习率，建议使用学习率调度器")
            
            # 根据激活函数类型添加建议
            if act_name == 'Sigmoid':
                specific_rec.append("考虑使用批归一化缓解梯度消失")
                specific_rec.append("适合输出层，隐藏层建议使用其他激活函数")
            elif act_name == 'Tanh':
                specific_rec.append("与批归一化配合使用效果更好")
            elif act_name == 'ReLU':
                specific_rec.append("使用He初始化而不是Xavier初始化")
                specific_rec.append("如果出现死亡ReLU问题，尝试LeakyReLU")
            elif act_name == 'LeakyReLU':
                specific_rec.append("α参数可适当调整（默认0.01通常效果良好）")
            
            if specific_rec:
                recommendations['specific_recommendations'][act_name] = specific_rec
        
        return recommendations
    
    def identify_limitations(self) -> List[str]:
        """识别实验的局限性"""
        limitations = [
            "实验在单一数据集（MNIST）上进行，结论可能不适用于其他数据集",
            "使用小型MLP架构，在更深的网络上表现可能不同",
            "未考虑激活函数与其他组件（如批归一化、残差连接）的交互",
            "实验仅进行了有限次重复，可能存在随机性影响",
            "未测试所有超参数组合（学习率、初始化方法等）",
            "未考虑计算效率的详细比较（内存占用、推理速度）",
            "梯度分析基于有限的数据点，可能不全面"
        ]
        
        return limitations
    
    def suggest_future_work(self) -> List[str]:
        """建议未来工作方向"""
        future_work = [
            "在更多样化的数据集上测试（CIFAR-10、ImageNet子集）",
            "测试更深层的网络架构（ResNet、DenseNet）",
            "研究激活函数与不同优化器的交互",
            "分析激活函数在迁移学习中的表现",
            "测试新的激活函数（Swish、Mish、GELU）",
            "进行超参数敏感性分析",
            "研究激活函数在自注意力机制中的表现",
            "分析激活函数对模型校准的影响",
            "测试激活函数在少样本学习场景的表现",
            "研究激活函数选择对对抗鲁棒性的影响"
        ]
        
        return future_work
    
    def save_analysis_report(self, analysis: Dict, filename: str = None):
        """保存分析报告"""
        if filename is None:
            timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
            filename = f"comprehensive_analysis_{timestamp}.json"
        
        Path(self.config.RESULT_SAVE_DIR).mkdir(exist_ok=True)
        
        # 转换numpy类型以便JSON序列化
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            else:
                return obj
        
        analysis_converted = convert_numpy_types(analysis)
        
        with open(f"{self.config.RESULT_SAVE_DIR}/{filename}", 'w', encoding='utf-8') as f:
            json.dump(analysis_converted, f, indent=2, ensure_ascii=False)
        
        print(f"分析报告已保存到: {self.config.RESULT_SAVE_DIR}/{filename}")
        
        # 同时生成简要文本报告
        self.generate_text_summary(analysis, filename.replace('.json', '_summary.txt'))
        
        return filename
    
    def generate_text_summary(self, analysis: Dict, filename: str):
        """生成文本格式的简要报告"""
        summary_lines = []
        
        summary_lines.append("=" * 80)
        summary_lines.append("激活函数比较实验 - 综合分析报告")
        summary_lines.append("=" * 80)
        summary_lines.append(f"生成时间: {analysis['timestamp']}")
        summary_lines.append(f"数据集: {analysis['dataset']}")
        summary_lines.append(f"模型架构: {analysis['model_architecture']}")
        summary_lines.append("")
        
        # 性能排名
        summary_lines.append("性能排名:")
        summary_lines.append("-" * 40)
        
        if 'convergence_analysis' in analysis:
            sorted_by_acc = sorted(analysis['convergence_analysis'].items(),
                                 key=lambda x: x[1]['best_accuracy'],
                                 reverse=True)
            
            for i, (act_name, info) in enumerate(sorted_by_acc, 1):
                summary_lines.append(f"{i}. {act_name}: {info['best_accuracy']:.2f}% "
                                   f"(收敛速度: {info['convergence_speed']}, "
                                   f"过拟合程度: {info['overfitting_level']})")
        
        summary_lines.append("")
        
        # 梯度分析
        summary_lines.append("梯度流动分析:")
        summary_lines.append("-" * 40)
        
        if 'gradient_flow_analysis' in analysis:
            for act_name, info in analysis['gradient_flow_analysis'].items():
                summary_lines.append(f"{act_name}:")
                summary_lines.append(f"  平均梯度大小: {info['mean_gradient']:.2e}")
                summary_lines.append(f"  梯度消失风险: {info['vanishing_gradient_risk']}")
                summary_lines.append(f"  梯度爆炸风险: {info['exploding_gradient_risk']}")
        
        summary_lines.append("")
        
        # 大模型预测准确性
        summary_lines.append("大模型预测准确性:")
        summary_lines.append("-" * 40)
        
        if 'llm_prediction_accuracy' in analysis:
            metrics = analysis['llm_prediction_accuracy']['accuracy_metrics']
            if metrics:
                summary_lines.append(f"总体准确率: {metrics['overall_accuracy']:.1f}%")
                summary_lines.append(f"部分正确率: {metrics['partial_accuracy']:.1f}%")
                summary_lines.append(f"正确预测: {', '.join(analysis['llm_prediction_accuracy']['correct_predictions'])}")
                summary_lines.append(f"错误预测: {', '.join(analysis['llm_prediction_accuracy']['incorrect_predictions'])}")
        
        summary_lines.append("")
        
        # 推荐
        summary_lines.append("推荐:")
        summary_lines.append("-" * 40)
        
        if 'recommendations' in analysis:
            rec = analysis['recommendations']
            summary_lines.append(f"最佳整体表现: {rec['best_overall']}")
            summary_lines.append(f"最快收敛: {rec['fastest_convergence']}")
            summary_lines.append(f"最稳定训练: {rec['most_stable']}")
            summary_lines.append(f"最少过拟合: {rec['least_overfitting']}")
            summary_lines.append(f"最容易调参: {rec['easiest_to_tune']}")
        
        summary_lines.append("")
        
        # 关键发现
        summary_lines.append("关键发现:")
        summary_lines.append("-" * 40)
        
        # 从统计检验中提取显著差异
        if 'statistical_tests' in analysis:
            significant_diffs = []
            for test_name, test_result in analysis['statistical_tests'].items():
                if test_result['t_test']['significant']:
                    act1, act2 = test_name.split('_vs_')
                    diff = test_result['mean_difference']
                    if diff > 0:
                        significant_diffs.append(f"{act1} 显著优于 {act2} (差异: {diff:.2f}%)")
                    else:
                        significant_diffs.append(f"{act2} 显著优于 {act1} (差异: {-diff:.2f}%)")
            
            if significant_diffs:
                summary_lines.extend(significant_diffs)
            else:
                summary_lines.append("未发现统计显著的性能差异")
        
        summary_lines.append("")
        
        # 局限性
        summary_lines.append("实验局限性:")
        summary_lines.append("-" * 40)
        if 'limitations' in analysis:
            for limitation in analysis['limitations']:
                summary_lines.append(f"• {limitation}")
        
        summary_lines.append("")
        
        # 保存文本报告
        with open(f"{self.config.RESULT_SAVE_DIR}/{filename}", 'w', encoding='utf-8') as f:
            f.write('\n'.join(summary_lines))
        
        print(f"文本摘要已保存到: {self.config.RESULT_SAVE_DIR}/{filename}")


def main():
    """测试分析模块"""
    from config import Config
    from evaluator import ComparisonAnalyzer
    
    config = Config()
    analyzer = ActivationFunctionAnalyzer(config)
    
    # 测试理论特性分析
    theoretical_props = analyzer.analyze_theoretical_properties()
    print("激活函数理论特性:")
    for act_name, props in theoretical_props.items():
        print(f"\n{act_name}:")
        print(f"  函数: {props['function']}")
        print(f"  范围: {props['range']}")
        print(f"  零中心化: {props['zero_centered']}")
    
    return analyzer


if __name__ == "__main__":
    main()