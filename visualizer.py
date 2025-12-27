"""结果可视化模块"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
import json
import re
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.backends.backend_pdf import PdfPages
import warnings
warnings.filterwarnings('ignore')

def setup_chinese_font():
    """设置中文字体支持"""
    try:
        # Windows系统
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'KaiTi', 'FangSong']
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
        return True
    except Exception as e:
        print(f"⚠ 中文字体设置失败，使用默认字体: {e}")
        return False

# 设置中文字体
setup_chinese_font()

# 设置图表样式
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
        
        ax.set_xlabel('训练轮数', fontsize=12)
        ax.set_ylabel('验证准确率 (%)', fontsize=12)
        ax.set_title('验证准确率对比', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. 训练损失对比
        ax = axes[0, 1]
        for act_name, result in results_dict.items():
            history = result['history']
            train_loss = history['train_loss']
            epochs = range(1, len(train_loss) + 1)
            ax.plot(epochs, train_loss, label=act_name, linewidth=2)
        
        ax.set_xlabel('训练轮数', fontsize=12)
        ax.set_ylabel('训练损失', fontsize=12)
        ax.set_title('训练损失对比', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. 训练准确率对比
        ax = axes[0, 2]
        for act_name, result in results_dict.items():
            history = result['history']
            train_acc = history['train_acc']
            epochs = range(1, len(train_acc) + 1)
            ax.plot(epochs, train_acc, label=act_name, linewidth=2)
        
        ax.set_xlabel('训练轮数', fontsize=12)
        ax.set_ylabel('训练准确率 (%)', fontsize=12)
        ax.set_title('训练准确率对比', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. 学习率变化
        ax = axes[1, 0]
        for act_name, result in results_dict.items():
            history = result['history']
            lr = history['learning_rate']
            epochs = range(1, len(lr) + 1)
            ax.plot(epochs, lr, label=act_name, linewidth=2)
        
        ax.set_xlabel('训练轮数', fontsize=12)
        ax.set_ylabel('学习率', fontsize=12)
        ax.set_title('学习率调度', fontsize=14, fontweight='bold')
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
        ax.set_xlabel('激活函数', fontsize=12)
        ax.set_ylabel('每轮平均训练时间 (秒)', fontsize=12)
        ax.set_title('计算效率', fontsize=14, fontweight='bold')
        
        # 在柱子上添加数值
        for bar, time_val in zip(bars, epoch_times):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                   f'{time_val:.2f}秒', ha='center', va='bottom', fontsize=10)
        
        # 6. 验证损失对比
        ax = axes[1, 2]
        for act_name, result in results_dict.items():
            history = result['history']
            val_loss = history['val_loss']
            epochs = range(1, len(val_loss) + 1)
            ax.plot(epochs, val_loss, label=act_name, linewidth=2)
        
        ax.set_xlabel('训练轮数', fontsize=12)
        ax.set_ylabel('验证损失', fontsize=12)
        ax.set_title('验证损失对比', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.suptitle('激活函数训练曲线对比', fontsize=16, y=1.02)
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
        
        bars1 = ax.bar(x - width/2, best_accs, width, label='最佳验证准确率', 
                      color='lightblue')
        bars2 = ax.bar(x + width/2, final_accs, width, label='最终验证准确率',
                      color='lightcoral')
        
        ax.set_xlabel('激活函数', fontsize=12)
        ax.set_ylabel('准确率 (%)', fontsize=12)
        ax.set_title('最佳 vs 最终验证准确率', fontsize=14, fontweight='bold')
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
            '激活函数': act_names,
            '收敛轮数': convergence_data
        }).set_index('激活函数')
        
        # 创建自定义颜色映射
        cmap = LinearSegmentedColormap.from_list(
            'convergence_cmap', 
            ['#4CAF50', '#FFEB3B', '#F44336']  # 绿->黄->红
        )
        
        im = ax.imshow([convergence_data], cmap=cmap, aspect='auto')
        ax.set_xticks(range(len(act_names)))
        ax.set_xticklabels(act_names, rotation=45, ha='right')
        ax.set_yticks([])
        ax.set_title('收敛速度热力图', fontsize=14, fontweight='bold')
        
        # 添加数值
        for i, val in enumerate(convergence_data):
            ax.text(i, 0, f' {val} 轮', ha='center', va='center', 
                   color='white' if val > np.mean(convergence_data) else 'black',
                   fontweight='bold')
        
        plt.colorbar(im, ax=ax, label='收敛轮数 (越小越快)')
        
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
        ax.set_xlabel('激活函数', fontsize=12)
        ax.set_ylabel('标准差 (最后5轮)', fontsize=12)
        ax.set_title('训练稳定性', fontsize=14, fontweight='bold')
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
            '准确率': [r['best_val_acc'] for r in results_dict.values()],
            '收敛速度': [1/c if c > 0 else 0 for c in convergence_data],  # 倒数，越大越好
            '稳定性': [1/(s+0.001) for s in stability_data],  # 倒数，越大越稳定
            '最终准确率': [r['final_val_acc'] for r in results_dict.values()]
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
        ax.set_title('综合性能雷达图', fontsize=14, fontweight='bold')
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax.grid(True)
        
        plt.suptitle('激活函数性能总结', fontsize=16, y=1.02)
        plt.tight_layout()
        plt.savefig(self.save_dir / filename, dpi=150, bbox_inches='tight')
        plt.show()
        
        return fig
    
    def plot_prediction_vs_reality(self, predictions_data, llm_predictions,
                                  filename='prediction_vs_reality.png'):
        """绘制预测vs实际对比图 - 完全修复版"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 提取数据
        act_names = list(predictions_data.keys())
        
        # 1. 收敛速度对比
        ax = axes[0, 0]
        predicted_speeds = []
        actual_speeds = []
        match_status_speed = []
        speed_names = []
        
        for act_name in act_names:
            if (act_name in predictions_data and 
                'convergence_speed' in predictions_data[act_name]):
                conv_data = predictions_data[act_name]['convergence_speed']
                
                # 获取预测的收敛速度
                pred_speed = conv_data.get('predicted', '中等')
                actual_speed = conv_data.get('actual', '中等')
                match = conv_data.get('match', False)
                
                # 将文本速度转换为数值用于绘图
                speed_map = {'快': 3, '中等': 2, '慢': 1, 'Fast': 3, 'Medium': 2, 'Slow': 1}
                
                predicted_speeds.append(speed_map.get(pred_speed, 2))
                actual_speeds.append(speed_map.get(actual_speed, 2))
                match_status_speed.append(match)
                speed_names.append(act_name)
        
        if predicted_speeds:
            x = np.arange(len(predicted_speeds))
            width = 0.35
            
            # 创建颜色数组：匹配为绿色，不匹配为红色
            pred_colors = ['#4CAF50' if m else '#F44336' for m in match_status_speed]
            actual_colors = ['#4CAF50' if m else '#F44336' for m in match_status_speed]
            
            bars1 = ax.bar(x - width/2, predicted_speeds, width, 
                          label='预测速度', color=pred_colors, alpha=0.7)
            bars2 = ax.bar(x + width/2, actual_speeds, width,
                          label='实际速度', color=actual_colors, alpha=0.7)
            
            ax.set_xlabel('激活函数', fontsize=12)
            ax.set_ylabel('收敛速度 (3=快, 2=中, 1=慢)', fontsize=12)
            ax.set_title('收敛速度: 预测 vs 实际', fontsize=14, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(speed_names, rotation=45)
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
            
            # 添加文本标签
            reverse_speed_map = {3: '快', 2: '中等', 1: '慢'}
            for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
                pred_text = reverse_speed_map.get(predicted_speeds[i], '中等')
                actual_text = reverse_speed_map.get(actual_speeds[i], '中等')
                
                ax.text(bar1.get_x() + bar1.get_width()/2, bar1.get_height(),
                       pred_text, ha='center', va='bottom', fontsize=9)
                
                ax.text(bar2.get_x() + bar2.get_width()/2, bar2.get_height(),
                       actual_text, ha='center', va='bottom', fontsize=9)
                
                # 在x轴下方添加匹配状态
                match_text = '✓' if match_status_speed[i] else '✗'
                ax.text(i, -0.2, match_text, ha='center', va='top', fontsize=12,
                       color='green' if match_status_speed[i] else 'red')
        else:
            ax.text(0.5, 0.5, '无收敛速度数据', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('收敛速度: 预测 vs 实际', fontsize=14, fontweight='bold')
        
        # 2. 准确率对比 - 修复版
        ax = axes[0, 1]
        
        predicted_accs = []
        actual_accs = []
        pred_lower = []
        pred_upper = []
        match_status_acc = []
        accuracy_names = []
        
        for act_name in act_names:
            if (act_name in predictions_data and 
                'accuracy' in predictions_data[act_name]):
                acc_data = predictions_data[act_name]['accuracy']
                
                # 获取预测准确率字符串
                pred_str = str(acc_data.get('predicted', '0%'))
                
                # 从字符串提取数值
                numbers = re.findall(r'\d+\.?\d*', pred_str)
                if numbers:
                    nums = [float(num) for num in numbers]
                    if len(nums) == 1:
                        # 单个数值
                        pred_val = nums[0]
                        lower = max(0, pred_val - 2)
                        upper = pred_val + 2
                    else:
                        # 范围值
                        lower = min(nums)
                        upper = max(nums)
                        pred_val = (lower + upper) / 2
                else:
                    # 默认值
                    pred_val = 85
                    lower = 83
                    upper = 87
                
                # 获取实际准确率
                actual_str = str(acc_data.get('actual', '0%'))
                actual_numbers = re.findall(r'\d+\.?\d*', actual_str)
                actual_val = float(actual_numbers[0]) if actual_numbers else 0
                
                # 获取匹配状态
                match = acc_data.get('match', False)
                
                predicted_accs.append(pred_val)
                actual_accs.append(actual_val)
                pred_lower.append(lower)
                pred_upper.append(upper)
                match_status_acc.append(match)
                accuracy_names.append(act_name)
        
        if predicted_accs:
            x = np.arange(len(predicted_accs))
            width = 0.35
            
            # 创建颜色数组
            pred_colors = ['#4CAF50' if m else '#F44336' for m in match_status_acc]
            actual_colors = ['#4CAF50' if m else '#F44336' for m in match_status_acc]
            
            # 绘制预测柱状图
            bars1 = ax.bar(x - width/2, predicted_accs, width, 
                          label='预测准确率', color=pred_colors, alpha=0.7)
            
            # 绘制实际柱状图
            bars2 = ax.bar(x + width/2, actual_accs, width,
                          label='实际准确率', color=actual_colors, alpha=0.7)
            
            # 添加预测范围的误差条 - 修复版本
            for i, (pred_val, lower, upper) in enumerate(zip(predicted_accs, pred_lower, pred_upper)):
                # 确保 lower <= pred_val <= upper
                if lower > pred_val:
                    lower = pred_val
                if upper < pred_val:
                    upper = pred_val
                
                # 计算误差条（确保非负）
                yerr_lower = max(0, pred_val - lower)
                yerr_upper = max(0, upper - pred_val)
                
                # 只有存在误差时才绘制误差条
                if yerr_lower > 0 or yerr_upper > 0:
                    ax.errorbar(x[i] - width/2, pred_val, 
                              yerr=[[yerr_lower], [yerr_upper]],
                              fmt='none', color='black', capsize=5, linewidth=1)
            
            ax.set_xlabel('激活函数', fontsize=12)
            ax.set_ylabel('准确率 (%)', fontsize=12)
            ax.set_title('准确率: 预测 vs 实际', fontsize=14, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(accuracy_names, rotation=45)
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
            
            # 添加数值标签
            for i, (bar, val) in enumerate(zip(bars1, predicted_accs)):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, height,
                       f'{val:.1f}%', ha='center', va='bottom', fontsize=9)
            
            for i, (bar, val) in enumerate(zip(bars2, actual_accs)):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, height,
                       f'{val:.2f}%', ha='center', va='bottom', fontsize=9)
                
                # 在x轴下方添加匹配状态
                match_text = '✓' if match_status_acc[i] else '✗'
                ax.text(i, -0.2, match_text, ha='center', va='top', fontsize=12,
                       color='green' if match_status_acc[i] else 'red')
        else:
            ax.text(0.5, 0.5, '无准确率数据', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('准确率: 预测 vs 实际', fontsize=14, fontweight='bold')
        
        # 3. 预测准确性热力图
        ax = axes[1, 0]
        
        match_data = []
        heatmap_names = []
        for act_name in act_names:
            if act_name in predictions_data:
                match = predictions_data[act_name].get('overall_match', False)
                match_data.append(1 if match else 0)
                heatmap_names.append(act_name)
        
        if match_data:
            cmap = LinearSegmentedColormap.from_list(
                'match_cmap', ['#F44336', '#4CAF50']
            )
            
            # 创建热力图
            if len(match_data) > 0:
                im = ax.imshow([match_data], cmap=cmap, aspect='auto', 
                              extent=[-0.5, len(match_data)-0.5, -0.5, 0.5])
                ax.set_xticks(range(len(heatmap_names)))
                ax.set_xticklabels(heatmap_names, rotation=45, ha='right')
                ax.set_yticks([])
                ax.set_title('整体预测准确性 (绿色=正确, 红色=错误)', 
                            fontsize=14, fontweight='bold')
                
                # 添加文本
                for i, match_val in enumerate(match_data):
                    text = '✓' if match_val == 1 else '✗'
                    ax.text(i, 0, text, ha='center', va='center', 
                           color='white', fontweight='bold', fontsize=14)
            else:
                ax.text(0.5, 0.5, '无匹配数据', ha='center', va='center', transform=ax.transAxes)
                ax.set_title('预测准确性', fontsize=14, fontweight='bold')
        else:
            ax.text(0.5, 0.5, '无匹配数据', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('预测准确性', fontsize=14, fontweight='bold')
        
        # 4. 预测偏差分析
        ax = axes[1, 1]
        
        errors = []
        error_names = []
        for i in range(min(len(actual_accs), len(predicted_accs))):
            if i < len(accuracy_names):
                error = actual_accs[i] - predicted_accs[i]
                errors.append(error)
                error_names.append(accuracy_names[i])
        
        if errors:
            colors = ['green' if e >= 0 else 'red' for e in errors]
            
            x_pos = np.arange(len(errors))
            bars = ax.bar(x_pos, errors, color=colors)
            
            ax.set_xlabel('激活函数', fontsize=12)
            ax.set_ylabel('实际 - 预测 (%)', fontsize=12)
            ax.set_title('预测偏差 (正数=低估, 负数=高估)', 
                        fontsize=14, fontweight='bold')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(error_names, rotation=45)
            ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            ax.grid(True, alpha=0.3, axis='y')
            
            # 添加数值标签
            for bar, error in zip(bars, errors):
                height = bar.get_height()
                va = 'bottom' if height >= 0 else 'top'
                ax.text(bar.get_x() + bar.get_width()/2, height,
                       f'{height:+.1f}%', ha='center', va=va, fontsize=9,
                       fontweight='bold')
            
            # 添加水平参考线
            if len(errors) > 0:
                mean_error = np.mean(errors)
                ax.axhline(y=mean_error, color='blue', linestyle='--', linewidth=1, 
                          alpha=0.5, label=f'平均偏差: {mean_error:+.1f}%')
                ax.legend()
        else:
            ax.text(0.5, 0.5, '无预测偏差数据', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('预测偏差分析', fontsize=14, fontweight='bold')
        
        plt.suptitle('大模型预测 vs 实际结果对比', fontsize=16, y=1.02)
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
            ax = axes[0, 0]
            ax.text(0.5, 0.5, '没有梯度信息可用', ha='center', va='center', transform=ax.transAxes)
            plt.suptitle('梯度分析', fontsize=16, y=1.02)
            plt.tight_layout()
            plt.savefig(self.save_dir / filename, dpi=150, bbox_inches='tight')
            plt.show()
            print("⚠ 没有梯度信息可用")
            return fig
        
        act_names = list(gradient_stats_all.keys())
        
        # 1. 平均梯度范数
        ax = axes[0, 0]
        mean_norms = [gradient_stats_all[name]['mean_norm'] for name in act_names]
        std_norms = [gradient_stats_all[name]['std_norm'] for name in act_names]
        
        bars = ax.bar(act_names, mean_norms, yerr=std_norms, 
                     capsize=5, color='skyblue', alpha=0.7)
        ax.set_xlabel('激活函数', fontsize=12)
        ax.set_ylabel('平均梯度范数', fontsize=12)
        ax.set_title('平均梯度大小', fontsize=14, fontweight='bold')
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
        ax.set_xlabel('激活函数', fontsize=12)
        ax.set_ylabel('梯度范数范围', fontsize=12)
        ax.set_title('梯度范数范围 (均值 ± 范围)', fontsize=14, fontweight='bold')
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
        
        ax.set_xlabel('训练检查点', fontsize=12)
        ax.set_ylabel('平均梯度范数', fontsize=12)
        ax.set_title('训练过程中的梯度范数', fontsize=14, fontweight='bold')
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
            
            ax.set_xlabel('激活函数', fontsize=12)
            ax.set_ylabel('梯度范数', fontsize=12)
            ax.set_title('梯度分布箱线图', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            ax.set_yscale('log')
        else:
            ax.text(0.5, 0.5, '无梯度分布数据', ha='center', va='center', transform=ax.transAxes)
        
        plt.suptitle('梯度分析', fontsize=16, y=1.02)
        plt.tight_layout()
        plt.savefig(self.save_dir / filename, dpi=150, bbox_inches='tight')
        plt.show()
        
        return fig
    
    def create_comprehensive_report(self, results_dict, llm_predictions, 
                                   comparison_results, filename='comprehensive_report.pdf'):
        """创建综合报告（多页PDF）"""
        try:
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
                
                # 添加文本总结页
                fig_text = plt.figure(figsize=(11, 8.5))
                fig_text.clf()
                text_ax = fig_text.add_subplot(111)
                text_ax.axis('off')
                
                # 生成文本总结
                summary = self._generate_text_summary(results_dict, comparison_results)
                
                # 添加文本到图表
                text_ax.text(0.05, 0.95, '激活函数比较实验 - 综合报告', 
                            fontsize=16, fontweight='bold', transform=text_ax.transAxes)
                text_ax.text(0.05, 0.90, '=' * 50, fontsize=10, transform=text_ax.transAxes)
                
                y_pos = 0.85
                for line in summary.split('\n'):
                    text_ax.text(0.05, y_pos, line, fontsize=10, transform=text_ax.transAxes)
                    y_pos -= 0.03
                    if y_pos < 0.05:
                        # 如果内容太多，创建新页面
                        pdf.savefig(fig_text, bbox_inches='tight')
                        fig_text = plt.figure(figsize=(11, 8.5))
                        fig_text.clf()
                        text_ax = fig_text.add_subplot(111)
                        text_ax.axis('off')
                        y_pos = 0.95
                
                pdf.savefig(fig_text, bbox_inches='tight')
                plt.close(fig_text)
            
            print(f"✓ 综合报告已保存到: {self.save_dir / filename}")
            
        except Exception as e:
            print(f"⚠ 创建PDF报告时出错: {e}")
            # 尝试创建文本报告
            self._generate_text_report(results_dict, comparison_results, 
                                      filename.replace('.pdf', '.txt'))
    
    def _generate_text_summary(self, results_dict, comparison_results):
        """生成文本总结"""
        summary_lines = []
        
        summary_lines.append("激活函数比较实验 - 综合报告")
        summary_lines.append("=" * 60)
        summary_lines.append("")
        
        # 1. 性能排名（按最佳验证准确率）
        summary_lines.append("1. 性能排名（按最佳验证准确率）：")
        summary_lines.append("-" * 40)
        
        sorted_by_acc = sorted(results_dict.items(), 
                              key=lambda x: x[1]['best_val_acc'], 
                              reverse=True)
        
        for i, (act_name, result) in enumerate(sorted_by_acc, 1):
            summary_lines.append(f"{i}. {act_name}: {result['best_val_acc']:.2f}% "
                               f"(最终: {result['final_val_acc']:.2f}%, "
                               f"最佳轮数: {result['best_epoch']})")
        
        summary_lines.append("")
        
        # 2. 收敛速度排名
        summary_lines.append("2. 收敛速度排名（达到最佳准确率90%所需轮数）：")
        summary_lines.append("-" * 40)
        
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
            summary_lines.append(f"{i}. {act_name}: {epochs} 轮")
        
        summary_lines.append("")
        
        # 3. 大模型预测准确性
        summary_lines.append("3. 大模型预测准确性分析：")
        summary_lines.append("-" * 40)
        
        if comparison_results:
            correct_predictions = 0
            total_predictions = 0
            
            for act_name in comparison_results:
                if comparison_results[act_name].get('overall_match', False):
                    correct_predictions += 1
                total_predictions += 1
            
            if total_predictions > 0:
                accuracy_rate = correct_predictions / total_predictions * 100
                summary_lines.append(f"总体预测准确率: {accuracy_rate:.1f}% "
                                   f"({correct_predictions}/{total_predictions})")
                
                # 预测不准确的激活函数
                incorrect_acts = []
                for act_name in comparison_results:
                    if not comparison_results[act_name].get('overall_match', False):
                        incorrect_acts.append(act_name)
                
                if incorrect_acts:
                    summary_lines.append(f"预测不准确的激活函数: {', '.join(incorrect_acts)}")
                else:
                    summary_lines.append("所有预测都准确！")
            else:
                summary_lines.append("没有预测数据")
        else:
            summary_lines.append("没有对比结果数据")
        
        summary_lines.append("")
        
        # 4. 关键发现
        summary_lines.append("4. 关键发现：")
        summary_lines.append("-" * 40)
        
        # 找出表现最好的激活函数
        best_activation = sorted_by_acc[0][0] if sorted_by_acc else "N/A"
        summary_lines.append(f"• 整体表现最佳的激活函数: {best_activation}")
        
        # 找出收敛最快的激活函数
        fastest_activation = sorted_by_speed[0][0] if sorted_by_speed else "N/A"
        summary_lines.append(f"• 收敛最快的激活函数: {fastest_activation}")
        
        # 计算训练时间
        total_times = {name: result['total_training_time'] 
                      for name, result in results_dict.items()}
        if total_times:
            fastest_training = min(total_times.items(), key=lambda x: x[1])[0]
            summary_lines.append(f"• 训练最快的激活函数: {fastest_training}")
        
        # 分析过拟合情况
        overfitting_gaps = {}
        for act_name, result in results_dict.items():
            train_acc = result['history']['train_acc'][-1] if result['history']['train_acc'] else 0
            val_acc = result['history']['val_acc'][-1] if result['history']['val_acc'] else 0
            overfitting_gaps[act_name] = train_acc - val_acc
        
        if overfitting_gaps:
            least_overfitting = min(overfitting_gaps.items(), key=lambda x: x[1])[0]
            summary_lines.append(f"• 过拟合最少的激活函数: {least_overfitting}")
        
        summary_lines.append("")
        summary_lines.append("＝" * 60)
        summary_lines.append("报告生成完成")
        
        return "\n".join(summary_lines)
    
    def _generate_text_report(self, results_dict, comparison_results, filename):
        """生成文本格式的报告"""
        summary = self._generate_text_summary(results_dict, comparison_results)
        
        with open(self.save_dir / filename, 'w', encoding='utf-8') as f:
            f.write(summary)
        
        print(f"✓ 文本报告已保存到: {self.save_dir / filename}")


def main():
    """测试可视化模块"""
    from config import Config
    
    config = Config()
    visualizer = ResultVisualizer(config)
    
    print("可视化模块测试完成")
    return visualizer


if __name__ == "__main__":
    main()