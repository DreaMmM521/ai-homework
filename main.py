"""主程序入口 - 激活函数比较实验"""
import os
import json
import numpy as np
import argparse
import sys
from pathlib import Path
import time
from datetime import datetime

import torch

# 添加当前目录到Python路径
sys.path.append('.')

from config import Config
from data_loader import DataLoaderManager
from models import ActivationFactory
from trainer import ExperimentRunner
from evaluator import ModelEvaluator, ComparisonAnalyzer
from visualizer import ResultVisualizer
from analyzer import ActivationFunctionAnalyzer
from llm_predictions import LLMPredictor

# 自定义JSON编码器，处理NumPy类型
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        else:
            return super(NumpyEncoder, self).default(obj)

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='激活函数比较实验')
    
    parser.add_argument('--mode', type=str, default='all',
                       choices=['all', 'train', 'evaluate', 'visualize', 'analyze'],
                       help='运行模式: all(全部), train(仅训练), evaluate(仅评估), visualize(仅可视化), analyze(仅分析)')
    
    parser.add_argument('--activations', type=str, nargs='+',
                       default=['ReLU', 'Sigmoid', 'Tanh', 'LeakyReLU'],
                       help='要测试的激活函数列表')
    
    parser.add_argument('--epochs', type=int, default=20,
                       help='训练轮数')
    
    parser.add_argument('--batch_size', type=int, default=64,
                       help='批大小')
    
    parser.add_argument('--lr', type=float, default=0.001,
                       help='学习率')
    
    parser.add_argument('--dataset', type=str, default='MNIST',
                       choices=['MNIST', 'CIFAR10'],
                       help='数据集')
    
    parser.add_argument('--repeats', type=int, default=3,
                       help='重复实验次数')
    
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子')
    
    parser.add_argument('--output_dir', type=str, default='results',
                       help='输出目录')
    
    parser.add_argument('--load_results', type=str, default=None,
                       help='加载已有结果文件进行分析')
    
    return parser.parse_args()

def setup_directories():
    """设置目录结构"""
    directories = ['data', 'models', 'results', 'plots']
    
    for dir_name in directories:
        Path(dir_name).mkdir(exist_ok=True)
    
    print("目录结构设置完成")

def run_experiment(config, args):
    """运行完整实验"""
    print("=" * 80)
    print("开始激活函数比较实验")
    print("=" * 80)
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"数据集: {config.DATASET_NAME}")
    print(f"激活函数: {', '.join(args.activations)}")
    print(f"训练轮数: {config.EPOCHS}")
    print(f"学习率: {config.LEARNING_RATE}")
    print(f"批大小: {config.BATCH_SIZE}")
    print("=" * 80)
    
    start_time = time.time()
    
    # 步骤1: 获取大模型预测
    print("\n[步骤1/6] 获取大模型预测...")
    llm_predictor = LLMPredictor()
    llm_predictions = llm_predictor.get_predictions()
    llm_filename = llm_predictor.save_predictions(llm_predictions)
    print(f"✓ 大模型预测已保存: {llm_filename}")
    
    # 步骤2: 加载数据
    print("\n[步骤2/6] 加载数据...")
    data_manager = DataLoaderManager(config)
    train_loader, val_loader, test_loader = data_manager.get_dataloaders()
    print("✓ 数据加载完成")
    
    # 步骤3: 训练模型
    print("\n[步骤3/6] 训练模型...")
    experiment_runner = ExperimentRunner(config)
    
    # 只训练指定的激活函数
    if args.activations:
        all_results = {}
        for act_name in args.activations:
            if act_name in config.ACTIVATIONS:
                result = experiment_runner.run_single_experiment(
                    act_name, train_loader, val_loader
                )
                all_results[act_name] = result
            else:
                print(f"警告: 激活函数 {act_name} 不在配置中，跳过")
    else:
        all_results = experiment_runner.run_all_experiments(
            train_loader, val_loader
        )
    
    print("✓ 模型训练完成")
    
    # 步骤4: 评估模型
    print("\n[步骤4/6] 评估模型...")
    test_results = {}
    
    for act_name in all_results.keys():
        print(f"\n评估 {act_name} 模型...")
        
        # 创建模型
        model = ActivationFactory.create_model(act_name, config)
        
        # 尝试加载最佳模型
        model_path = f"results/models/best_model_{act_name}.pth"
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            print(f"  加载模型: {model_path}")
        else:
            print(f"  警告: 未找到 {act_name} 的已保存模型")
            # 尝试查找其他可能的位置
            import glob
            possible_files = glob.glob(f"results/models/*{act_name}*.pth")
            if possible_files:
                # 使用最新的文件
                latest_file = max(possible_files, key=os.path.getmtime)
                checkpoint = torch.load(latest_file)
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    model.load_state_dict(checkpoint)
                print(f"  从 {latest_file} 加载模型")
        
        # 评估
        evaluator = ModelEvaluator(model, config)
        test_result = evaluator.evaluate(test_loader)
        test_results[act_name] = test_result
        
        print(f"  测试准确率: {test_result['test_accuracy']:.2f}%")
    
    print("✓ 模型评估完成")
    
    # 步骤5: 对比分析
    print("\n[步骤5/6] 结果对比分析...")
    
    # 加载LLM预测
    llm_predictions_loaded = llm_predictor.load_predictions(llm_filename)
    
    # 对比分析
    comparison_analyzer = ComparisonAnalyzer(config)
    comparison_results = comparison_analyzer.generate_comparison_report(
        all_results, llm_predictions_loaded
    )
    
    # 保存对比结果（使用NumpyEncoder解决序列化问题）
    comparison_filename = f"comparison_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(f"{config.RESULT_SAVE_DIR}/{comparison_filename}", 'w') as f:
        json.dump(comparison_results, f, indent=2, cls=NumpyEncoder)
    
    print(f"✓ 对比分析完成，结果保存到: {comparison_filename}")
    
    # 步骤6: 可视化
    print("\n[步骤6/6] 生成可视化图表...")
    visualizer = ResultVisualizer(config)
    
    # 训练曲线
    visualizer.plot_training_curves(all_results, 'training_curves_comparison.png')
    
    # 性能总结
    visualizer.plot_performance_summary(all_results, 'performance_summary.png')
    
    # 预测vs实际
    visualizer.plot_prediction_vs_reality(
        comparison_results['predictions_vs_reality'],
        llm_predictions_loaded,
        'prediction_vs_reality.png'
    )
    
    # 梯度分析
    visualizer.plot_gradient_analysis(all_results, 'gradient_analysis.png')
    
    # 综合报告
    visualizer.create_comprehensive_report(
        all_results,
        llm_predictions_loaded,
        comparison_results['predictions_vs_reality'],
        'comprehensive_report.pdf'
    )
    
    print("✓ 可视化完成")
    
    # 步骤7: 深入分析
    print("\n[步骤7/7] 进行深入分析...")
    analyzer = ActivationFunctionAnalyzer(config)
    
    comprehensive_analysis = analyzer.generate_comprehensive_analysis(
        all_results, llm_predictions_loaded
    )
    
    analysis_filename = analyzer.save_analysis_report(comprehensive_analysis)
    
    print(f"✓ 深入分析完成，报告保存到: {analysis_filename}")
    
    # 计算总运行时间
    total_time = time.time() - start_time
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = int(total_time % 60)
    
    print("\n" + "=" * 80)
    print("实验完成!")
    print("=" * 80)
    print(f"总运行时间: {hours:02d}:{minutes:02d}:{seconds:02d}")
    print(f"结果保存在: {config.RESULT_SAVE_DIR}/")
    print(f"图表保存在: {config.PLOT_SAVE_DIR}/")
    print("=" * 80)
    
    # 打印简要总结
    print("\n简要总结:")
    print("-" * 40)
    
    # 性能排名
    if 'experiment_summary' in comparison_results:
        print("性能排名 (最佳验证准确率):")
        for act_name, result in comparison_results['experiment_summary']['ranking_by_accuracy'].items():
            print(f"  {act_name}: {result['best_val_acc']:.2f}%")
    
    # 预测准确性
    if 'llm_prediction_accuracy' in comprehensive_analysis:
        metrics = comprehensive_analysis['llm_prediction_accuracy']['accuracy_metrics']
        if metrics:
            print(f"\n大模型预测准确率: {metrics['overall_accuracy']:.1f}%")
    
    return {
        'all_results': all_results,
        'test_results': test_results,
        'comparison_results': comparison_results,
        'analysis': comprehensive_analysis
    }

def evaluate_existing_results(args):
    """评估已有结果"""
    print(f"加载已有结果: {args.load_results}")
    
    config = Config()
    
    # 加载结果
    with open(args.load_results, 'r') as f:
        existing_results = json.load(f)
    
    # 加载LLM预测
    llm_predictor = LLMPredictor()
    llm_predictions = llm_predictor.get_predictions()
    
    # 进行对比分析
    comparison_analyzer = ComparisonAnalyzer(config)
    comparison_results = comparison_analyzer.generate_comparison_report(
        existing_results, llm_predictions
    )
    
    # 可视化
    visualizer = ResultVisualizer(config)
    visualizer.plot_prediction_vs_reality(
        comparison_results['predictions_vs_reality'],
        llm_predictions,
        'prediction_vs_reality_existing.png'
    )
    
    # 深入分析
    analyzer = ActivationFunctionAnalyzer(config)
    comprehensive_analysis = analyzer.generate_comprehensive_analysis(
        existing_results, llm_predictions
    )
    
    analysis_filename = analyzer.save_analysis_report(comprehensive_analysis)
    
    print(f"分析完成，报告保存到: {analysis_filename}")
    
    return comprehensive_analysis

def main():
    """主函数"""
    args = parse_arguments()
    
    # 设置配置
    config = Config()
    
    # 更新配置参数
    if args.epochs:
        config.EPOCHS = args.epochs
    if args.batch_size:
        config.BATCH_SIZE = args.batch_size
    if args.lr:
        config.LEARNING_RATE = args.lr
    if args.dataset:
        config.DATASET_NAME = args.dataset
    if args.repeats:
        config.NUM_REPEATS = args.repeats
    if args.seed:
        config.SEED = args.seed
    if args.output_dir:
        config.RESULT_SAVE_DIR = args.output_dir
        config.MODEL_SAVE_DIR = f"{args.output_dir}/models"
        config.PLOT_SAVE_DIR = f"{args.output_dir}/plots"
    
    # 设置目录
    setup_directories()
    
    # 根据模式执行
    if args.mode == 'train' or args.mode == 'all':
        # 导入torch（在确定需要训练时）
        import torch
        
        if args.load_results:
            print("警告: --load_results 参数在训练模式下被忽略")
        
        results = run_experiment(config, args)
    elif args.mode == 'evaluate':
        if not args.load_results:
            print("错误: 评估模式需要指定 --load_results 参数")
            sys.exit(1)
        
        evaluate_existing_results(args)
    elif args.mode == 'visualize':
        print("可视化模式需要已有结果文件，请使用 --load_results 参数")
    elif args.mode == 'analyze':
        if not args.load_results:
            print("错误: 分析模式需要指定 --load_results 参数")
            sys.exit(1)
        
        evaluate_existing_results(args)
    else:
        print(f"未知模式: {args.mode}")
        sys.exit(1)

if __name__ == "__main__":
    main()