"""模型评估模块"""

import torch
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import json
from pathlib import Path

class ModelEvaluator:
    """模型评估器"""
    
    def __init__(self, model, config, device=None):
        self.model = model
        self.config = config
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
            
        self.model.to(self.device)
        self.model.eval()
    
    def evaluate(self, test_loader):
        """在测试集上评估模型"""
        all_predictions = []
        all_targets = []
        test_loss = 0.0
        correct = 0
        total = 0
        
        criterion = torch.nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                
                # 计算损失
                test_loss += criterion(output, target).item()
                
                # 获取预测
                pred = output.argmax(dim=1, keepdim=False)
                
                # 收集结果
                all_predictions.extend(pred.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                
                # 统计准确率
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        
        test_loss /= len(test_loader)
        test_accuracy = 100. * correct / total
        
        # 计算混淆矩阵
        cm = confusion_matrix(all_targets, all_predictions)
        
        # 计算每个类别的准确率
        class_accuracy = cm.diagonal() / cm.sum(axis=1) * 100
        
        # 分类报告
        report = classification_report(
            all_targets, all_predictions,
            target_names=[str(i) for i in range(10)],
            output_dict=True
        )
        
        results = {
            'test_loss': test_loss,
            'test_accuracy': test_accuracy,
            'confusion_matrix': cm.tolist(),
            'class_accuracy': class_accuracy.tolist(),
            'classification_report': report,
            'total_samples': total,
            'correct_samples': correct
        }
        
        return results
    
    def analyze_predictions(self, test_loader, num_samples=10):
        """分析预测样本"""
        self.model.eval()
        sample_analysis = []
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(test_loader):
                if len(sample_analysis) >= num_samples:
                    break
                    
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                probabilities = torch.softmax(output, dim=1)
                
                for i in range(min(len(target), num_samples - len(sample_analysis))):
                    sample = {
                        'true_label': target[i].item(),
                        'predicted_label': output[i].argmax().item(),
                        'confidence': probabilities[i].max().item(),
                        'top3_predictions': [
                            {'label': j, 'prob': probabilities[i][j].item()}
                            for j in probabilities[i].argsort(descending=True)[:3]
                        ]
                    }
                    sample_analysis.append(sample)
        
        return sample_analysis
    
    def compute_gradient_statistics(self, train_loader):
        """计算梯度统计信息"""
        self.model.train()
        gradient_norms = []
        gradient_means = []
        gradient_stds = []
        
        # 只用一个batch计算
        data, target = next(iter(train_loader))
        data, target = data.to(self.device), target.to(self.device)
        
        # 前向传播
        output = self.model(data)
        loss = torch.nn.functional.cross_entropy(output, target)
        
        # 反向传播
        self.model.zero_grad()
        loss.backward()
        
        # 收集梯度信息
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grad = param.grad.data.cpu().numpy().flatten()
                gradient_norms.append(np.linalg.norm(grad))
                gradient_means.append(np.mean(grad))
                gradient_stds.append(np.std(grad))
        
        gradient_stats = {
            'mean_norm': np.mean(gradient_norms),
            'std_norm': np.std(gradient_norms),
            'mean_gradient': np.mean(gradient_means),
            'std_gradient': np.mean(gradient_stds),
            'gradient_norms': gradient_norms,
            'gradient_means': gradient_means,
            'gradient_stds': gradient_stds
        }
        
        self.model.eval()
        return gradient_stats


class ComparisonAnalyzer:
    """对比分析器：比较不同激活函数的性能"""
    
    def __init__(self, config):
        self.config = config
    
    def load_results(self, results_dir='results'):
        """加载所有实验结果"""
        results = {}
        result_files = list(Path(results_dir).glob('result_*.json'))
        
        for file in result_files:
            # 从文件名提取激活函数名
            parts = file.stem.split('_')
            if len(parts) >= 2:
                activation_name = parts[1]
                
                with open(file, 'r') as f:
                    data = json.load(f)
                    results[activation_name] = data
        
        return results
    
    def compare_performance(self, results):
        """比较性能指标"""
        comparison = {}
        
        for act_name, result in results.items():
            comparison[act_name] = {
                'best_val_acc': result.get('best_val_acc', 0),
                'final_val_acc': result.get('final_val_acc', 0),
                'best_epoch': result.get('best_epoch', 0),
                'convergence_speed': self._calculate_convergence_speed(
                    result.get('history', {}).get('val_acc', [])
                ),
                'training_stability': self._calculate_stability(
                    result.get('history', {}).get('val_acc', [])
                ),
                'total_time': result.get('total_training_time', 0)
            }
        
        # 排序
        sorted_by_accuracy = sorted(
            comparison.items(),
            key=lambda x: x[1]['best_val_acc'],
            reverse=True
        )
        
        sorted_by_speed = sorted(
            comparison.items(),
            key=lambda x: x[1]['convergence_speed']
        )
        
        return {
            'comparison': comparison,
            'ranking_by_accuracy': dict(sorted_by_accuracy),
            'ranking_by_speed': dict(sorted_by_speed)
        }
    
    def _calculate_convergence_speed(self, val_acc_history):
        """计算收敛速度（达到最佳准确率90%所需的epoch数）"""
        if not val_acc_history:
            return float('inf')
        
        best_acc = max(val_acc_history)
        target_acc = 0.9 * best_acc
        
        for i, acc in enumerate(val_acc_history):
            if acc >= target_acc:
                return i + 1  # epoch从1开始
        
        return len(val_acc_history)
    
    def _calculate_stability(self, val_acc_history):
        """计算训练稳定性（最后5个epoch准确率的标准差）"""
        if len(val_acc_history) < 5:
            return float('inf')
        
        last_5 = val_acc_history[-5:]
        return np.std(last_5)
    
    def generate_comparison_report(self, results, llm_predictions):
        """生成对比报告"""
        comparison = self.compare_performance(results)
        
        report = {
            'experiment_summary': comparison,
            'llm_predictions': llm_predictions,
            'predictions_vs_reality': self._compare_with_predictions(
                comparison, llm_predictions
            ),
            'key_findings': self._extract_key_findings(
                comparison, llm_predictions
            )
        }
        
        return report
    
    def _compare_with_predictions(self, comparison, predictions):
        """与大模型预测对比"""
        comparison_results = {}
        
        for act_name in comparison['comparison'].keys():
            if act_name in predictions:
                exp_result = comparison['comparison'][act_name]
                llm_pred = predictions[act_name]['predicted_performance']
                
                # 收敛速度对比
                conv_speed_map = {
                    '快': 1,
                    '中等': 2, 
                    '慢': 3
                }
                
                actual_speed = '快' if exp_result['convergence_speed'] <= 5 else \
                              '中等' if exp_result['convergence_speed'] <= 10 else '慢'
                
                speed_match = actual_speed == llm_pred['convergence_speed']
                
                # 准确率对比
                actual_acc_level = '高' if exp_result['best_val_acc'] >= 95 else \
                                 '中' if exp_result['best_val_acc'] >= 90 else '低'
                
                # 从预测字符串提取准确率级别
                pred_acc_str = llm_pred['final_accuracy']
                if '高' in pred_acc_str:
                    pred_acc_level = '高'
                elif '中' in pred_acc_str:
                    pred_acc_level = '中'
                else:
                    pred_acc_level = '低'
                
                acc_match = actual_acc_level == pred_acc_level
                
                comparison_results[act_name] = {
                    'convergence_speed': {
                        'predicted': llm_pred['convergence_speed'],
                        'actual': actual_speed,
                        'match': speed_match,
                        'actual_epochs': exp_result['convergence_speed']
                    },
                    'accuracy': {
                        'predicted': llm_pred['final_accuracy'],
                        'actual': f"{exp_result['best_val_acc']:.2f}%",
                        'match': acc_match,
                        'accuracy_level': actual_acc_level
                    },
                    'overall_match': speed_match and acc_match
                }
        
        return comparison_results
    
    def _extract_key_findings(self, comparison, predictions):
        """提取关键发现"""
        findings = []
        comp_results = self._compare_with_predictions(comparison, predictions)
        
        # 统计预测准确性
        total = len(comp_results)
        correct = sum(1 for result in comp_results.values() 
                     if result['overall_match'])
        accuracy_rate = correct / total * 100
        
        findings.append(f"大模型预测总体准确率: {accuracy_rate:.1f}% ({correct}/{total})")
        
        # 分析每个激活函数
        for act_name, result in comp_results.items():
            if not result['overall_match']:
                findings.append(
                    f"{act_name}: 预测不准确。"
                    f"预测收敛速度'{result['convergence_speed']['predicted']}'，"
                    f"实际'{result['convergence_speed']['actual']}'。"
                    f"预测准确率'{result['accuracy']['predicted']}'，"
                    f"实际'{result['accuracy']['actual']}'。"
                )
        
        # 性能排名
        accuracy_rank = list(comparison['ranking_by_accuracy'].keys())
        findings.append(f"性能排名（从高到低）: {', '.join(accuracy_rank)}")
        
        speed_rank = list(comparison['ranking_by_speed'].keys())
        findings.append(f"收敛速度排名（从快到慢）: {', '.join(speed_rank)}")
        
        return findings


def main():
    """测试评估模块"""
    from config import Config
    from models import ActivationFactory
    from data_loader import DataLoaderManager
    
    config = Config()
    data_manager = DataLoaderManager(config)
    _, _, test_loader = data_manager.get_dataloaders()
    
    # 创建模型
    model = ActivationFactory.create_model('ReLU', config)
    
    # 评估模型
    evaluator = ModelEvaluator(model, config)
    results = evaluator.evaluate(test_loader)
    
    print(f"测试准确率: {results['test_accuracy']:.2f}%")
    print(f"测试损失: {results['test_loss']:.4f}")
    
    return results


if __name__ == "__main__":
    main()