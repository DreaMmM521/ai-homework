"""结果可视化模块"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
import json
from matplotlib.colors import LinearSegmentedColormap
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
sns.set_palette("husl")

class ResultVisualizer:
    """结果可视化器"""
    
    def __init__(self, config, save_dir='plots'):
        self.config = config
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
    
    def plot_training_curves(self, results_dict, filename='training_curves.png'):
        """绘制训练曲线对比图"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        # 1. 验证准确率对比
        ax = axes[0, 0]
        for act_name, result in results_dict.items():
            history = result['history']
            val_acc = history['val_acc']
            epochs = range(1, len(val_acc) + 1)
            ax.plot(epochs, val_acc, label=act_name, linewidth=2)
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Validation Accuracy (%)', fontsize=12)
        ax.set_title('Validation Accuracy Comparison', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. 训练损失对比
        ax = axes[0, 1]
        for act_name, result in results_dict.items():
            history = result['history']
            train_loss = history['train_loss']
            epochs = range(1, len(train_loss) + 1)
            ax.plot(epochs, train_loss, label=act_name, linewidth=2)
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Training Loss', fontsize=12)
        ax.set_title('Training Loss Comparison', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. 训练准确率对比
        ax = axes[0, 2]
        for act_name, result in results_dict.items():
            history = result['history']
            train_acc = history['train_acc']
            epochs = range(1, len(train_acc) + 1)
            ax.plot(epochs, train_acc, label=act_name, linewidth=2)
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Training Accuracy (%)', fontsize=12)
        ax.set_title('Training Accuracy Comparison', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. 学习率变化
        ax = axes[1, 0]
        for act_name, result in results_dict.items():
            history = result['history']
            lr = history['learning_rate']
            epochs = range(1, len(lr) + 1)
            ax.plot(epochs, lr, label=act_name, linewidth=2)
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Learning Rate', fontsize=12)
        ax.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        
        # 5. 每个epoch的训练时间
        ax = axes[1, 1]
        epoch_times = []
        labels = []
        for act_name, result in results_dict.items():
            history = result['history']
            avg_time = np.mean(history['epoch_time'])
            epoch_times.append(avg_time)
            labels.append(act_name)
        
        bars = ax.bar(labels, epoch_times)
        ax.set_xlabel('Activation Function', fontsize=12)
        ax.set_ylabel('Average Time per Epoch (s)', fontsize=12)
        ax.set_title('Computational Efficiency', fontsize=14, fontweight='bold')
        
        # 在柱子上添加数值
        for bar, time_val in zip(bars, epoch_times):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                   f'{time_val:.2f}s', ha='center', va='bottom', fontsize=10)
        
        # 6. 验证损失对比
        ax = axes[1, 2]
        for act_name, result in results_dict.items():
            history = result['history']
            val_loss = history['val_loss']
            epochs = range(1, len(val_loss) + 1)
            ax.plot(epochs, val_loss, label=act_name, linewidth=2)
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Validation Loss', fontsize=12)
        ax.set_title('Validation Loss Comparison', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / filename, dpi=150, bbox_inches='tight')
        plt.show()
        
        return fig
    
    def plot_performance_summary(self, results_dict, filename='performance_summary.png'):
        """绘制性能总结图"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. 最终验证准确率柱状图
        ax = axes[0, 0]
        act_names = []
        final_accs = []
        best_accs = []
        
        for act_name, result in results_dict.items():
            act_names.append(act_name)
            final_accs.append(result['final_val_acc'])
            best_accs.append(result['best_val_acc'])
        
        x = np.arange(len(act_names))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, best_accs, width, label='Best Validation Acc', 
                      color='lightblue')
        bars2 = ax.bar(x + width/2, final_accs, width, label='Final Validation Acc',
                      color='lightcoral')
        
        ax.set_xlabel('Activation Function', fontsize=12)
        ax.set_ylabel('Accuracy (%)', fontsize=12)
        ax.set_title('Best vs Final Validation Accuracy', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(act_names)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # 添加数值标签
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, height,
                       f'{height:.2f}%', ha='center', va='bottom', fontsize=9)
        
        # 2. 收敛速度热力图
        ax = axes[0, 1]
        
        # 计算收敛速度（达到最佳准确率90%的epoch）
        convergence_data = []
        for act_name, result in results_dict.items():
            val_acc = result['history']['val_acc']
            best_acc = max(val_acc)
            target_acc = 0.9 * best_acc
            
            for i, acc in enumerate(val_acc):
                if acc >= target_acc:
                    convergence_data.append(i + 1)
                    break
            else:
                convergence_data.append(len(val_acc))
        
        # 创建热力图数据
        heatmap_data = pd.DataFrame({
            'Activation': act_names,
            'Convergence Epoch': convergence_data
        }).set_index('Activation')
        
        # 创建自定义颜色映射
        cmap = LinearSegmentedColormap.from_list(
            'convergence_cmap', 
            ['#4CAF50', '#FFEB3B', '#F44336']  # 绿->黄->红
        )
        
        im = ax.imshow([convergence_data], cmap=cmap, aspect='auto')
        ax.set_xticks(range(len(act_names)))
        ax.set_xticklabels(act_names, rotation=45, ha='right')
        ax.set_yticks([])
        ax.set_title('Convergence Speed Heatmap', fontsize=14, fontweight='bold')
        
        # 添加数值
        for i, val in enumerate(convergence_data):
            ax.text(i, 0, f' {val} epochs', ha='center', va='center', 
                   color='white' if val > np.mean(convergence_data) else 'black',
                   fontweight='bold')
        
        plt.colorbar(im, ax=ax, label='Convergence Epoch (lower is faster)')
        
        # 3. 训练稳定性（最后5个epoch准确率的标准差）
        ax = axes[1, 0]
        stability_data = []
        for act_name, result in results_dict.items():
            val_acc = result['history']['val_acc']
            if len(val_acc) >= 5:
                last_5 = val_acc[-5:]
                stability = np.std(last_5)
            else:
                stability = np.std(val_acc)
            stability_data.append(stability)
        
        bars = ax.bar(act_names, stability_data, color='lightgreen')
        ax.set_xlabel('Activation Function', fontsize=12)
        ax.set_ylabel('Standard Deviation (Last 5 Epochs)', fontsize=12)
        ax.set_title('Training Stability', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # 添加数值标签
        for bar, stability in zip(bars, stability_data):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height,
                   f'{stability:.3f}', ha='center', va='bottom', fontsize=9)
        
        # 4. 综合评分雷达图
        ax = axes[1, 1]
        
        # 计算综合评分（归一化）
        metrics = {
            'Accuracy': [r['best_val_acc'] for r in results_dict.values()],
            'Convergence': [1/c if c > 0 else 0 for c in convergence_data],  # 倒数，越大越好
            'Stability': [1/(s+0.001) for s in stability_data],  # 倒数，越大越稳定
            'Final Acc': [r['final_val_acc'] for r in results_dict.values()]
        }
        
        # 归一化到0-1
        normalized_metrics = {}
        for metric_name, values in metrics.items():
            min_val, max_val = min(values), max(values)
            if max_val > min_val:
                normalized = [(v - min_val) / (max_val - min_val) for v in values]
            else:
                normalized = [0.5 for _ in values]
            normalized_metrics[metric_name] = normalized
        
        # 设置雷达图
        categories = list(normalized_metrics.keys())
        N = len(categories)
        
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # 闭合图形
        
        # 绘制每个激活函数
        for idx, act_name in enumerate(act_names):
            values = [normalized_metrics[metric][idx] for metric in categories]
            values += values[:1]  # 闭合图形
            
            ax.plot(angles, values, linewidth=2, label=act_name)
            ax.fill(angles, values, alpha=0.1)
        
        # 设置标签
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=11)
        ax.set_ylim(0, 1)
        ax.set_title('Overall Performance Radar Chart', fontsize=14, fontweight='bold')
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax.grid(True)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / filename, dpi=150, bbox_inches='tight')
        plt.show()
        
        return fig
    
    def plot_prediction_vs_reality(self, comparison_results, llm_predictions,
                                  filename='prediction_vs_reality.png'):
        """绘制预测vs实际对比图"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 提取数据
        act_names = list(comparison_results.keys())
        
        # 1. 收敛速度对比
        ax = axes[0, 0]
        predicted_speeds = []
        actual_speeds_numeric = []
        actual_speeds_text = []
        
        for act_name in act_names:
            if act_name in llm_predictions:
                pred = llm_predictions[act_name]['predicted_performance']['convergence_speed']
                actual = comparison_results[act_name]['convergence_speed']['actual_epochs']
                
                # 将文本速度转换为数值
                speed_map = {'快': 1, '中等': 2, '慢': 3}
                actual_text = '快' if actual <= 5 else '中等' if actual <= 10 else '慢'
                
                predicted_speeds.append(speed_map.get(pred, 2))
                actual_speeds_numeric.append(actual)
                actual_speeds_text.append(speed_map.get(actual_text, 2))
        
        x = np.arange(len(act_names))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, predicted_speeds, width, 
                      label='Predicted (Rank)', color='lightblue')
        bars2 = ax.bar(x + width/2, actual_speeds_numeric, width,
                      label='Actual (Epochs)', color='lightcoral')
        
        ax.set_xlabel('Activation Function', fontsize=12)
        ax.set_ylabel('Convergence Speed', fontsize=12)
        ax.set_title('Predicted vs Actual Convergence Speed', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(act_names, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # 添加文本标签
        for i, (bar1, bar2, act_name) in enumerate(zip(bars1, bars2, act_names)):
            pred_text = list(speed_map.keys())[list(speed_map.values()).index(predicted_speeds[i])]
            ax.text(bar1.get_x() + bar1.get_width()/2, bar1.get_height(),
                   pred_text, ha='center', va='bottom', fontsize=9)
            
            ax.text(bar2.get_x() + bar2.get_width()/2, bar2.get_height(),
                   f'{actual_speeds_numeric[i]} epochs', ha='center', 
                   va='bottom', fontsize=9)
        
        # 2. 准确率对比
        ax = axes[0, 1]
        
        predicted_accs = []
        actual_accs = []
        
        for act_name in act_names:
            if act_name in llm_predictions:
                pred_str = llm_predictions[act_name]['predicted_performance']['final_accuracy']
                # 从字符串提取数值范围
                import re
                numbers = re.findall(r'\d+\.?\d*', pred_str)
                if numbers:
                    pred_val = sum(float(n) for n in numbers) / len(numbers)
                else:
                    pred_val = 85  # 默认值
                
                actual_val = float(comparison_results[act_name]['accuracy']['actual'].replace('%', ''))
                
                predicted_accs.append(pred_val)
                actual_accs.append(actual_val)
        
        x = np.arange(len(act_names))
        bars1 = ax.bar(x - width/2, predicted_accs, width, 
                      label='Predicted Range (Mid)', color='lightblue')
        bars2 = ax.bar(x + width/2, actual_accs, width,
                      label='Actual', color='lightcoral')
        
        ax.set_xlabel('Activation Function', fontsize=12)
        ax.set_ylabel('Accuracy (%)', fontsize=12)
        ax.set_title('Predicted vs Actual Accuracy', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(act_names, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # 添加误差条表示预测范围
        for i, act_name in enumerate(act_names):
            if act_name in llm_predictions:
                pred_str = llm_predictions[act_name]['predicted_performance']['final_accuracy']
                numbers = re.findall(r'\d+\.?\d*', pred_str)
                if len(numbers) >= 2:
                    lower, upper = float(numbers[0]), float(numbers[1])
                    ax.errorbar(i - width/2, predicted_accs[i], 
                              yerr=[[predicted_accs[i] - lower], [upper - predicted_accs[i]]],
                              fmt='none', color='black', capsize=5)
        
        # 3. 预测准确性热力图
        ax = axes[1, 0]
        
        match_data = []
        for act_name in act_names:
            if act_name in comparison_results:
                match = comparison_results[act_name]['overall_match']
                match_data.append(1 if match else 0)
        
        cmap = LinearSegmentedColormap.from_list(
            'match_cmap', ['#F44336', '#4CAF50']  # 红->绿
        )
        
        im = ax.imshow([match_data], cmap=cmap, aspect='auto')
        ax.set_xticks(range(len(act_names)))
        ax.set_xticklabels(act_names, rotation=45, ha='right')
        ax.set_yticks([])
        ax.set_title('Prediction Accuracy (Green=Correct, Red=Incorrect)', 
                    fontsize=14, fontweight='bold')
        
        # 添加文本
        for i, match_val in enumerate(match_data):
            text = '✓ Correct' if match_val == 1 else '✗ Incorrect'
            ax.text(i, 0, text, ha='center', va='center', 
                   color='white', fontweight='bold', fontsize=10)
        
        # 4. 预测偏差分析
        ax = axes[1, 1]
        
        errors = []
        for act_name in act_names:
            if act_name in comparison_results:
                actual_acc = float(comparison_results[act_name]['accuracy']['actual'].replace('%', ''))
                pred_str = llm_predictions[act_name]['predicted_performance']['final_accuracy']
                numbers = re.findall(r'\d+\.?\d*', pred_str)
                if numbers:
                    pred_mid = sum(float(n) for n in numbers) / len(numbers)
                    error = actual_acc - pred_mid
                else:
                    error = 0
                errors.append(error)
        
        colors = ['green' if e >= 0 else 'red' for e in errors]
        bars = ax.bar(act_names, errors, color=colors)
        
        ax.set_xlabel('Activation Function', fontsize=12)
        ax.set_ylabel('Actual - Predicted (%)', fontsize=12)
        ax.set_title('Prediction Bias (Positive=Underestimated)', 
                    fontsize=14, fontweight='bold')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.grid(True, alpha=0.3, axis='y')
        
        # 添加数值标签
        for bar, error in zip(bars, errors):
            height = bar.get_height()
            va = 'bottom' if height >= 0 else 'top'
            ax.text(bar.get_x() + bar.get_width()/2, height,
                   f'{height:+.1f}%', ha='center', va=va, fontsize=9,
                   fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.save_dir / filename, dpi=150, bbox_inches='tight')
        plt.show()
        
        return fig
    
    def plot_gradient_analysis(self, results_dict, filename='gradient_analysis.png'):
        """绘制梯度分析图"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 收集梯度信息
        gradient_stats_all = {}
        
        for act_name, result in results_dict.items():
            if 'gradient_stats' in result:
                gradient_stats = result['gradient_stats']
                if gradient_stats:
                    # 提取梯度范数
                    norms = []
                    for stat in gradient_stats:
                        if 'grad_info' in stat:
                            for param_name, grad_info in stat['grad_info'].items():
                                if 'mean' in grad_info:
                                    norms.append(abs(grad_info['mean']))
                    
                    if norms:
                        gradient_stats_all[act_name] = {
                            'mean_norm': np.mean(norms),
                            'std_norm': np.std(norms),
                            'max_norm': np.max(norms),
                            'min_norm': np.min(norms)
                        }
        
        if not gradient_stats_all:
            print("没有梯度信息可用")
            return
        
        act_names = list(gradient_stats_all.keys())
        
        # 1. 平均梯度范数
        ax = axes[0, 0]
        mean_norms = [gradient_stats_all[name]['mean_norm'] for name in act_names]
        std_norms = [gradient_stats_all[name]['std_norm'] for name in act_names]
        
        bars = ax.bar(act_names, mean_norms, yerr=std_norms, 
                     capsize=5, color='skyblue', alpha=0.7)
        ax.set_xlabel('Activation Function', fontsize=12)
        ax.set_ylabel('Average Gradient Norm', fontsize=12)
        ax.set_title('Average Gradient Magnitude', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # 添加数值标签
        for bar, mean_val in zip(bars, mean_norms):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                   f'{mean_val:.2e}', ha='center', va='bottom', fontsize=9)
        
        # 2. 梯度范围（最大值/最小值）
        ax = axes[0, 1]
        max_norms = [gradient_stats_all[name]['max_norm'] for name in act_names]
        min_norms = [gradient_stats_all[name]['min_norm'] for name in act_names]
        
        # 使用误差条表示范围
        x_pos = range(len(act_names))
        ax.errorbar(x_pos, mean_norms, 
                   yerr=[np.array(mean_norms) - np.array(min_norms),
                         np.array(max_norms) - np.array(mean_norms)],
                   fmt='o', capsize=5, color='red', alpha=0.6)
        
        ax.scatter(x_pos, mean_norms, s=100, color='blue', zorder=5)
        ax.set_xlabel('Activation Function', fontsize=12)
        ax.set_ylabel('Gradient Norm Range', fontsize=12)
        ax.set_title('Gradient Norm Range (Mean ± Range)', fontsize=14, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(act_names)
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        
        # 3. 梯度变化趋势
        ax = axes[1, 0]
        for act_name in act_names:
            if act_name in results_dict:
                gradient_stats = results_dict[act_name]['gradient_stats']
                if gradient_stats:
                    # 提取每个检查点的平均梯度
                    checkpoints = []
                    avg_grads = []
                    
                    for i, stat in enumerate(gradient_stats[:20]):  # 只取前20个检查点
                        if 'grad_info' in stat:
                            grad_means = []
                            for param_name, grad_info in stat['grad_info'].items():
                                if 'mean' in grad_info:
                                    grad_means.append(abs(grad_info['mean']))
                            
                            if grad_means:
                                checkpoints.append(i)
                                avg_grads.append(np.mean(grad_means))
                    
                    if checkpoints:
                        ax.plot(checkpoints, avg_grads, label=act_name, linewidth=2)
        
        ax.set_xlabel('Training Checkpoint', fontsize=12)
        ax.set_ylabel('Average Gradient Norm', fontsize=12)
        ax.set_title('Gradient Norm During Training', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        
        # 4. 梯度分布箱线图
        ax = axes[1, 1]
        
        all_grad_data = []
        labels = []
        
        for act_name in act_names:
            if act_name in results_dict:
                gradient_stats = results_dict[act_name]['gradient_stats']
                if gradient_stats:
                    # 收集所有梯度值
                    grad_values = []
                    for stat in gradient_stats:
                        if 'grad_info' in stat:
                            for param_name, grad_info in stat['grad_info'].items():
                                if 'mean' in grad_info:
                                    grad_values.append(abs(grad_info['mean']))
                    
                    if grad_values:
                        # 采样避免数据过多
                        if len(grad_values) > 100:
                            grad_values = np.random.choice(grad_values, 100, replace=False)
                        all_grad_data.append(grad_values)
                        labels.append(act_name)
        
        if all_grad_data:
            bp = ax.boxplot(all_grad_data, labels=labels, patch_artist=True)
            
            # 设置颜色
            colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow']
            for patch, color in zip(bp['boxes'], colors[:len(labels)]):
                patch.set_facecolor(color)
            
            ax.set_xlabel('Activation Function', fontsize=12)
            ax.set_ylabel('Gradient Norm', fontsize=12)
            ax.set_title('Gradient Distribution Boxplot', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            ax.set_yscale('log')
        
        plt.tight_layout()
        plt.savefig(self.save_dir / filename, dpi=150, bbox_inches='tight')
        plt.show()
        
        return fig
    
    def create_comprehensive_report(self, results_dict, llm_predictions, 
                                   comparison_results, filename='comprehensive_report.pdf'):
        """创建综合报告（多页PDF）"""
        from matplotlib.backends.backend_pdf import PdfPages
        
        with PdfPages(self.save_dir / filename) as pdf:
            # 第1页：训练曲线对比
            fig1 = self.plot_training_curves(results_dict)
            pdf.savefig(fig1, bbox_inches='tight')
            plt.close(fig1)
            
            # 第2页：性能总结
            fig2 = self.plot_performance_summary(results_dict)
            pdf.savefig(fig2, bbox_inches='tight')
            plt.close(fig2)
            
            # 第3页：预测vs实际
            fig3 = self.plot_prediction_vs_reality(comparison_results, llm_predictions)
            pdf.savefig(fig3, bbox_inches='tight')
            plt.close(fig3)
            
            # 第4页：梯度分析
            fig4 = self.plot_gradient_analysis(results_dict)
            pdf.savefig(fig4, bbox_inches='tight')
            plt.close(fig4)
            
            # 添加文本页
            fig_text, ax = plt.subplots(figsize=(11, 8))
            ax.axis('off')
            
            # 创建文本摘要
            summary_text = self._generate_text_summary(results_dict, llm_predictions, comparison_results)
            
            # 将文本添加到图中
            ax.text(0.1, 0.95, '实验总结报告', fontsize=24, fontweight='bold', 
                   transform=ax.transAxes)
            ax.text(0.1, 0.85, '='*50, fontsize=12, transform=ax.transAxes)
            
            y_pos = 0.75
            for line in summary_text.split('\n'):
                ax.text(0.1, y_pos, line, fontsize=10, transform=ax.transAxes,
                       verticalalignment='top')
                y_pos -= 0.04
            
            pdf.savefig(fig_text, bbox_inches='tight')
            plt.close(fig_text)
        
        print(f"综合报告已保存到: {self.save_dir / filename}")
    
    def _generate_text_summary(self, results_dict, llm_predictions, comparison_results):
        """生成文本摘要"""
        summary = []
        
        # 性能排名
        summary.append("性能排名（基于最佳验证准确率）：")
        sorted_by_acc = sorted(results_dict.items(), 
                              key=lambda x: x[1]['best_val_acc'], 
                              reverse=True)
        
        for i, (act_name, result) in enumerate(sorted_by_acc, 1):
            summary.append(f"  {i}. {act_name}: {result['best_val_acc']:.2f}%")
        
        summary.append("")
        
        # 收敛速度排名
        summary.append("收敛速度排名（达到最佳准确率90%所需epoch数）：")
        
        convergence_times = {}
        for act_name, result in results_dict.items():
            val_acc = result['history']['val_acc']
            best_acc = max(val_acc)
            target_acc = 0.9 * best_acc
            
            for epoch, acc in enumerate(val_acc, 1):
                if acc >= target_acc:
                    convergence_times[act_name] = epoch
                    break
            else:
                convergence_times[act_name] = len(val_acc)
        
        sorted_by_speed = sorted(convergence_times.items(), key=lambda x: x[1])
        
        for i, (act_name, epochs) in enumerate(sorted_by_speed, 1):
            summary.append(f"  {i}. {act_name}: {epochs} epochs")
        
        summary.append("")
        
        # 预测准确性
        summary.append("大模型预测准确性分析：")
        
        correct_predictions = 0
        total_predictions = 0
        
        for act_name in comparison_results:
            if comparison_results[act_name]['overall_match']:
                correct_predictions += 1
            total_predictions += 1
        
        accuracy_rate = correct_predictions / total_predictions * 100
        summary.append(f"  总体预测准确率: {accuracy_rate:.1f}% ({correct_predictions}/{total_predictions})")
        
        summary.append("")
        summary.append("预测不准确的激活函数：")
        
        for act_name in comparison_results:
            if not comparison_results[act_name]['overall_match']:
                conv = comparison_results[act_name]['convergence_speed']
                acc = comparison_results[act_name]['accuracy']
                summary.append(f"  - {act_name}:")
                summary.append(f"    收敛速度: 预测'{conv['predicted']}', 实际'{conv['actual']}'")
                summary.append(f"    准确率: 预测'{acc['predicted']}', 实际'{acc['actual']}'")
        
        summary.append("")
        summary.append("关键发现：")
        
        # 找出表现最好的激活函数
        best_activation = sorted_by_acc[0][0]
        summary.append(f"  1. 整体表现最佳的激活函数: {best_activation}")
        
        # 找出收敛最快的激活函数
        fastest_activation = sorted_by_speed[0][0]
        summary.append(f"  2. 收敛最快的激活函数: {fastest_activation}")
        
        # 分析梯度特性
        if 'gradient_stats' in next(iter(results_dict.values())):
            summary.append("  3. 梯度特性分析:")
            
            # 计算平均梯度大小
            avg_gradients = {}
            for act_name, result in results_dict.items():
                if 'gradient_stats' in result and result['gradient_stats']:
                    grad_norms = []
                    for stat in result['gradient_stats']:
                        if 'grad_info' in stat:
                            for param_info in stat['grad_info'].values():
                                if 'mean' in param_info:
                                    grad_norms.append(abs(param_info['mean']))
                    
                    if grad_norms:
                        avg_gradients[act_name] = np.mean(grad_norms)
            
            if avg_gradients:
                min_grad_act = min(avg_gradients.items(), key=lambda x: x[1])[0]
                max_grad_act = max(avg_gradients.items(), key=lambda x: x[1])[0]
                summary.append(f"    - 梯度最大的激活函数: {max_grad_act}")
                summary.append(f"    - 梯度最小的激活函数: {min_grad_act}")
        
        return "\n".join(summary)


def main():
    """测试可视化模块"""
    from config import Config
    from evaluator import ComparisonAnalyzer
    
    config = Config()
    visualizer = ResultVisualizer(config)
    
    # 加载示例数据
    analyzer = ComparisonAnalyzer(config)
    results = analyzer.load_results()
    llm_predictions = {}
    
    # 创建示例图表
    if results:
        fig1 = visualizer.plot_training_curves(results)
        fig2 = visualizer.plot_performance_summary(results)
    
    return visualizer


if __name__ == "__main__":
    main()